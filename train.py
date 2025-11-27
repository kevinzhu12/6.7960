import torch
import os
import numpy as np
from PIL import Image
from typing import Tuple, Any
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from peft import LoraConfig, get_peft_model
# from trl import DDPOTrainer, DDPOConfig, DDPOStableDiffusionPipeline
from trl.trl.trainer.ddpo_trainer import DDPOTrainer
from trl.trl.trainer.ddpo_config import DDPOConfig
from trl.trl.models import DDPOStableDiffusionPipeline
from reward import CamouflageRewardFunction
from accelerate import Accelerator
from dataset import prepare_ddpo_dataset, load_dataset

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

def main():
    # Disable mixed precision to avoid VAE dtype issues with img2img
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=1,
        project_dir="./ddpo_output"
    )
    
    # Load pipeline in float32 to avoid dtype mismatches with VAE
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    ).to(accelerator.device)

    # load dataset
    accelerator.print("Loading dataset...")
    dataset_root = "datasets/COD10K-v3"
    dataset = load_dataset(dataset_root)
    train_dataset_raw = dataset['train']

    # TODO: remove this after testing
    train_dataset_raw = train_dataset_raw.select(range(5))

    # get label names
    label_names = train_dataset_raw.features['label'].names
    labels = [f"a photo of a {label}" for label in label_names]
    accelerator.print(f"Found {len(label_names)} labels: {label_names[:5]}...")

    # prepare ddpo dataset
    accelerator.print("Preparing DDPO dataset...")
    train_dataset = prepare_ddpo_dataset(
        dataset=train_dataset_raw,
        labels=labels
    )
    accelerator.print(f"Prepared {len(train_dataset)} samples")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v"],
        lora_dropout=0.00,
        bias="none",
    )

    accelerator.print("Applying LoRA adapters...")
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    accelerator.print(f"✓ LoRA applied: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    
    # Wrap pipeline for DDPO (minimal wrapper)
    accelerator.print("Wrapping pipeline for DDPO...")
    class Img2ImgDDPOPipeline(DDPOStableDiffusionPipeline):
        def __init__(self, pipeline):
            self.sd_pipeline = pipeline
            self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config)
            self.sd_pipeline.vae.requires_grad_(False)
            self.sd_pipeline.text_encoder.requires_grad_(False)
            self._current_images = []  # Store current batch images
            self.use_lora = True  # Flag for checkpoint saving/loading
        
        def __call__(self, prompt_embeds=None, negative_prompt_embeds=None, 
                     num_inference_steps=10, guidance_scale=7.5, eta=0.0, **kwargs):
            """Custom img2img pipeline with proper latent tracking for DDPO."""
            from trl.trl.models.modeling_sd_base import DDPOPipelineOutput, scheduler_step
            from torchvision import transforms
            
            # Get images from stored batch
            if not self._current_images:
                raise ValueError("Initial images not set. Call prompt_function first.")
            
            device = self.sd_pipeline.unet.device
            # Use strength=1.0 to ensure we run all timesteps (required by DDPO)
            # This means we start from mostly noise but conditioned on the encoded image
            strength = 1.0
            
            # 1. Preprocess input images to tensors
            to_tensor = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
            image_tensors = []
            for img in self._current_images:
                if isinstance(img, torch.Tensor):
                    image_tensors.append(img)
                else:
                    image_tensors.append(to_tensor(img))
            images = torch.stack(image_tensors).to(device=device, dtype=torch.float32)
            # Normalize to [-1, 1] for VAE
            images = 2.0 * images - 1.0
            
            batch_size = images.shape[0]
            
            # 2. Encode images to latent space
            with torch.no_grad():
                init_latents = self.sd_pipeline.vae.encode(images).latent_dist.sample()
                init_latents = init_latents * self.sd_pipeline.vae.config.scaling_factor
            
            # 3. Setup timesteps based on strength
            self.sd_pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
            # Calculate starting timestep based on strength
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = self.sd_pipeline.scheduler.timesteps[t_start:]
            
            # 4. Add noise to latents
            noise = torch.randn_like(init_latents)
            latents = self.sd_pipeline.scheduler.add_noise(init_latents, noise, timesteps[:1])
            
            # 5. Setup guidance
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance and negative_prompt_embeds is not None:
                prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
            else:
                prompt_embeds_input = prompt_embeds
            
            # 6. Denoising loop with latent tracking
            all_latents = [latents]
            all_log_probs = []
            
            for t in timesteps:
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.sd_pipeline.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict noise
                with torch.no_grad():
                    noise_pred = self.sd_pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds_input,
                        return_dict=False,
                    )[0]
                
                # Apply classifier-free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Scheduler step with log prob tracking
                scheduler_output = scheduler_step(self.sd_pipeline.scheduler, noise_pred, t, latents, eta)
                latents = scheduler_output.latents
                log_prob = scheduler_output.log_probs
                
                all_latents.append(latents)
                all_log_probs.append(log_prob)
            
            # 7. Decode latents to images
            with torch.no_grad():
                images_out = self.sd_pipeline.vae.decode(latents / self.sd_pipeline.vae.config.scaling_factor, return_dict=False)[0]
            
            # Clear stored images after use
            self._current_images = []
            
            return DDPOPipelineOutput(
                images=images_out,
                latents=all_latents,
                log_probs=all_log_probs,
            )
        def scheduler_step(self, *args, **kwargs):
            from trl.trl.models.modeling_sd_base import scheduler_step
            return scheduler_step(self.sd_pipeline.scheduler, *args, **kwargs)
        def set_progress_bar_config(self, **kwargs): pass
        def get_trainable_layers(self):
            return self.sd_pipeline.unet
        @property
        def autocast(self):
            import contextlib
            return contextlib.nullcontext
        @property
        def unet(self): return self.sd_pipeline.unet
        @property
        def vae(self): return self.sd_pipeline.vae
        @property
        def tokenizer(self): return self.sd_pipeline.tokenizer
        @property
        def scheduler(self): return self.sd_pipeline.scheduler
        @property
        def text_encoder(self): return self.sd_pipeline.text_encoder
        
        def save_checkpoint(self, models, weights, output_dir):
            """Save LoRA checkpoint"""
            from peft.utils import get_peft_model_state_dict
            from trl.trl.models.sd_utils import convert_state_dict_to_diffusers
            
            if len(models) != 1:
                raise ValueError("Given how the trainable params were set, this should be of length 1")
            
            if self.use_lora and hasattr(models[0], "peft_config") and getattr(models[0], "peft_config", None) is not None:
                state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(models[0]))
                self.sd_pipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)
            else:
                raise ValueError(f"Unknown model type {type(models[0])}")
        
        def load_checkpoint(self, models, input_dir):
            """Load LoRA checkpoint"""
            if len(models) != 1:
                raise ValueError("Given how the trainable params were set, this should be of length 1")
            
            if self.use_lora:
                lora_state_dict, network_alphas = self.sd_pipeline.lora_state_dict(
                    input_dir, weight_name="pytorch_lora_weights.safetensors"
                )
                self.sd_pipeline.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=models[0])
            else:
                raise ValueError(f"Unknown model type {type(models[0])}")
    
    ddpo_pipeline = Img2ImgDDPOPipeline(pipe)

    # Create prompt function - DDPO expects () -> (prompt, metadata)
    accelerator.print("Setting up prompt and reward functions...")
    prompt_iterator = iter(train_dataset)
    def prompt_function():
        nonlocal prompt_iterator
        try:
            sample = next(prompt_iterator)
            # Store image in pipeline for img2img (append to batch list)
            ddpo_pipeline._current_images.append(sample['image'])
            return sample['prompt'], sample['metadata']
        except StopIteration:
            prompt_iterator = iter(train_dataset)
            sample = next(prompt_iterator)
            ddpo_pipeline._current_images.append(sample['image'])
            return sample['prompt'], sample['metadata']

    # Wrap reward function - DDPO expects (tensor, tuple[str], tuple[Any]) -> tensor
    reward_fn_instance = CamouflageRewardFunction(
        clip_model_name="openai/clip-vit-base-patch32",
        device=str(accelerator.device),
        use_lpips_penalty=True,
        lpips_weight=0.1,
        normalize_rewards=True,
    )
    
    def reward_function(images_tensor: torch.Tensor, prompts: Tuple[str], metadata: Tuple[Any]) -> torch.Tensor:
        # Convert tensor to PIL
        pil_images = []
        for i in range(images_tensor.shape[0]):
            img = images_tensor[i].cpu()
            img = (img + 1.0) / 2.0  # [-1,1] -> [0,1]
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img))
        # Call reward function
        rewards = reward_fn_instance(pil_images, list(prompts), list(metadata))
        return torch.tensor(rewards, device=images_tensor.device, dtype=torch.float32), {}

    # Track rewards over training
    reward_history = []
    
    ddpo_config = DDPOConfig(
        logdir="./ddpo_output",
        num_epochs=5,  # Increased for better training visibility
        train_batch_size=4,
        train_gradient_accumulation_steps=1,
        mixed_precision="no",  # Disable mixed precision to avoid VAE dtype issues with img2img
        train_learning_rate=1e-4,
        train_max_grad_norm=1.0,
        num_checkpoint_limit=3,
        sample_batch_size=4,
        sample_num_batches_per_epoch=2,
        save_freq=1,
        # Wandb configuration
        log_with="wandb",
        tracker_project_name="ddpo-camouflage",  # Project name (required)
        tracker_kwargs={
            "wandb_project": "ddpo-camouflage",  # Must match tracker_project_name
            "wandb_name": "camouflage-enhancement",  # Run name (optional)
        },
    )
    
    accelerator.print("\n" + "="*60)
    accelerator.print("DDPO Training Configuration:")
    accelerator.print("="*60)
    accelerator.print(f"  Epochs: {ddpo_config.num_epochs}")
    accelerator.print(f"  Train batch size: {ddpo_config.train_batch_size}")
    accelerator.print(f"  Sample batch size: {ddpo_config.sample_batch_size}")
    accelerator.print(f"  Samples per epoch: {ddpo_config.sample_batch_size * ddpo_config.sample_num_batches_per_epoch}")
    accelerator.print(f"  Learning rate: {ddpo_config.train_learning_rate}")
    accelerator.print(f"  Mixed precision: {ddpo_config.mixed_precision}")
    accelerator.print(f"  Output directory: {ddpo_config.logdir}")
    accelerator.print(f"  Logging: {ddpo_config.log_with}")
    if ddpo_config.log_with == "wandb":
        accelerator.print(f"  Wandb project: {ddpo_config.tracker_kwargs.get('wandb_project', 'N/A')}")
    accelerator.print("="*60 + "\n")

    # Image logging hook for training progress
    def image_samples_hook(image_data, global_step, accelerate_logger):
        """Log images and rewards during training"""
        if image_data:
            # Log the last batch
            images, prompts, _, rewards, _ = image_data[-1]
            avg_reward = rewards.mean().item()
            min_reward = rewards.min().item()
            max_reward = rewards.max().item()
            std_reward = rewards.std().item()
            
            # Track reward history
            reward_history.append({
                'step': global_step,
                'avg': avg_reward,
                'min': min_reward,
                'max': max_reward,
                'std': std_reward,
            })
            
            accelerator.print(f"\n[Step {global_step}] Avg: {avg_reward:.4f} | Min: {min_reward:.4f} | Max: {max_reward:.4f}")
            
            # Log to wandb if available
            if accelerate_logger is not None:
                import wandb
                # Log metrics
                accelerate_logger.log({
                    "reward/mean": avg_reward,
                    "reward/min": min_reward,
                    "reward/max": max_reward,
                    "reward/std": std_reward,
                    "reward/range": max_reward - min_reward,
                }, step=global_step)
                
                # Log sample images every 5 steps
                if global_step % 5 == 0:
                    wandb_images = []
                    prompt_metadata = image_data[-1][2]  # Get metadata (contains original images)
                    for i in range(min(2, len(images))):  # Log first 2 images
                        # Get generated image
                        gen_img = images[i].cpu()
                        gen_img = ((gen_img + 1.0) / 2.0 * 255).clamp(0, 255).byte()
                        gen_img = gen_img.permute(1, 2, 0).numpy()
                        gen_img_pil = Image.fromarray(gen_img)
                        
                        # Get original image from metadata
                        original_img = prompt_metadata[i]['original_image']
                        if isinstance(original_img, Image.Image):
                            if original_img.size != (512, 512):
                                original_img = original_img.resize((512, 512))
                        else:
                            original_img = gen_img_pil  # Fallback
                        
                        # Create side-by-side comparison
                        comparison = Image.new('RGB', (1024, 512))
                        comparison.paste(original_img, (0, 0))
                        comparison.paste(gen_img_pil, (512, 0))
                        
                        wandb_images.append(wandb.Image(
                            comparison,
                            caption=f"Step {global_step} | Reward: {rewards[i].item():.4f}\nOriginal (left) → Enhanced (right)\nPrompt: {prompts[i][:50]}..."
                        ))
                    
                    accelerate_logger.log({
                        "samples": wandb_images,
                    }, step=global_step)
            
            # Save sample images every 5 steps (local backup)
            if global_step % 5 == 0:
                sample_dir = f'outputs/samples/step_{global_step}'
                os.makedirs(sample_dir, exist_ok=True)
                for i in range(min(2, len(images))):  # Save first 2 images
                    img = images[i].cpu()
                    img = ((img + 1.0) / 2.0 * 255).clamp(0, 255).byte()
                    img = img.permute(1, 2, 0).numpy()
                    Image.fromarray(img).save(f'{sample_dir}/img_{i}_r{rewards[i].item():.2f}.png')
                accelerator.print(f"  → Saved samples to {sample_dir}/")

    trainer = DDPOTrainer(
        config=ddpo_config,
        reward_function=reward_function,
        prompt_function=prompt_function,
        sd_pipeline=ddpo_pipeline,
        image_samples_hook=image_samples_hook,
    )
    
    accelerator.print("✓ Trainer initialized")
    accelerator.print("\n" + "="*60)
    accelerator.print("STARTING TRAINING")
    accelerator.print("="*60 + "\n")
    
    trainer.train()
    
    accelerator.print("\n" + "="*60)
    accelerator.print("TRAINING COMPLETE")
    accelerator.print("="*60)

    if accelerator.is_main_process:
        accelerator.print("\nSaving model and logs...")
        os.makedirs("outputs", exist_ok=True)
        pipe.unet.save_pretrained("outputs/lora_weights")
        accelerator.print("✓ LoRA weights saved to outputs/lora_weights/")

        stats = reward_fn_instance.get_stats()
        accelerator.print("\nFinal Reward Statistics:")
        accelerator.print(f"  Mean reward: {stats.get('mean_reward', 0):.4f}")
        accelerator.print(f"  Std reward: {stats.get('std_reward', 0):.4f}")
        accelerator.print(f"  Min reward: {stats.get('min_reward', 0):.4f}")
        accelerator.print(f"  Max reward: {stats.get('max_reward', 0):.4f}")
        
        # Show reward progression
        if reward_history:
            accelerator.print("\nReward Progression:")
            for r in reward_history:
                accelerator.print(f"  Step {r['step']:3d}: avg={r['avg']:.4f}")
            # Show trend
            if len(reward_history) >= 2:
                first_avg = reward_history[0]['avg']
                last_avg = reward_history[-1]['avg']
                change = last_avg - first_avg
                accelerator.print(f"\n  Trend: {'+' if change >= 0 else ''}{change:.4f} (first → last)")

        # save training logs
        with open("outputs/training_logs.txt", "w") as f:
            f.write(f"Training Configuration:\n")
            f.write(f"  Epochs: {ddpo_config.num_epochs}\n")
            f.write(f"  Train batch size: {ddpo_config.train_batch_size}\n")
            f.write(f"  Sample batch size: {ddpo_config.sample_batch_size}\n")
            f.write(f"  Learning rate: {ddpo_config.train_learning_rate}\n")
            f.write(f"\nTraining complete\n")
            f.write(f"\nReward function statistics:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
        accelerator.print("✓ Training logs saved to outputs/training_logs.txt")
        accelerator.print("\n" + "="*60)
        accelerator.print("ALL DONE!")
        accelerator.print("="*60)


if __name__ == "__main__":
    main()