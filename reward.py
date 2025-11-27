import torch
from transformers import pipeline
from PIL import Image
from typing import List, Dict, Any
import lpips
import torchvision.transforms as transforms
import numpy as np

class CamouflageRewardFunction:
    """
    Reward function for camouflaged animal enhancement.
    
    Reward = CLIP classifier confidence for correct class - optional LPIPS drift penalty
    
    This encourages the model to:
    1. Make animals more detectable (higher CLIP confidence)
    2. Minimize perceptual changes (lower LPIPS penalty)
    """
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        use_lpips_penalty: bool = True,
        lpips_weight: float = 0.1,
        device: str = "cuda",
        normalize_rewards: bool = True,
    ):
        """
        Args:
            clip_model_name: CLIP model for zero-shot classification
            use_lpips_penalty: Whether to penalize perceptual drift from original
            lpips_weight: Weight for LPIPS penalty (higher = more conservative)
            device: Device to run on
            normalize_rewards: Whether to normalize rewards (recommended for DDPO)
        """
        self.device = device
        self.use_lpips_penalty = use_lpips_penalty
        self.lpips_weight = lpips_weight
        self.normalize_rewards = normalize_rewards
        
        # Initialize CLIP classifier
        print(f"Loading CLIP model: {clip_model_name}")
        self.clip = pipeline(
            task="zero-shot-image-classification",
            model=clip_model_name,
            dtype=torch.bfloat16,
            use_fast=True,
            device=0 if device == "cuda" else -1
        )
        
        # Optional: LPIPS for perceptual similarity
        if use_lpips_penalty:
            print("Loading LPIPS model...")
            self.lpips_model = lpips.LPIPS(net='alex').to(torch.device(device))
            self.lpips_model.eval()
            
            # Transform for LPIPS
            self.lpips_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        # Track reward statistics for normalization
        self.reward_history = []
    
    def __call__(
        self,
        images: List[Image.Image],
        prompts: List[str],
        metadata: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Compute rewards for a batch of enhanced images.
        
        Args:
            images: List of PIL Images (enhanced images from diffusion model)
            prompts: List of prompts (empty strings in your case)
            metadata: List of dicts with keys:
                - 'original_image': PIL Image (original camouflaged image)
                - 'true_label': str like "a photo of a BatFish"
                - 'label_names': List[str] of all possible label strings
        
        Returns:
            List of reward values (floats), one per image
        """
        batch_size = len(images)
        rewards = []
        
        for i in range(batch_size):
            enhanced_img = images[i]
            meta = metadata[i]
            original_img = meta['original_image']
            true_label_text = meta['true_label']
            label_names = meta['label_names']
            
            # Step 1: Get CLIP classifier confidence for true label
            confidence = self._get_clip_confidence(
                enhanced_img, 
                true_label_text, 
                label_names
            )
            
            # Step 2: Compute base reward (classifier confidence)
            reward = confidence
            
            # Step 3: Optional: subtract perceptual drift penalty
            if self.use_lpips_penalty:
                lpips_loss = self._compute_lpips(original_img, enhanced_img)
                reward -= self.lpips_weight * lpips_loss

            # import pdb; pdb.set_trace()
            
            rewards.append(reward)
        
        # Step 4: Normalize rewards (important for stable DDPO training)
        if self.normalize_rewards:
            rewards = self._normalize_rewards(rewards)
        
        # Track for monitoring
        self.reward_history.extend(rewards)
        if len(self.reward_history) > 1000:  # Keep last 1000 for stats
            self.reward_history = self.reward_history[-1000:]
        
        return rewards
    
    def _get_clip_confidence(
        self, 
        image: Image.Image, 
        true_label: str, 
        label_names: List[str]
    ) -> float:
        """
        Get CLIP confidence score for the true label.
        
        Returns:
            Confidence score between 0 and 1
        """
        # CLIP expects list of label strings
        likelihoods = self.clip(image, label_names)
        
        # Find score for true label
        confidence = 0.0
        for pred in likelihoods:
            if pred['label'] == true_label:
                confidence = pred['score']
                break
        
        return confidence
    
    def _compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Compute LPIPS perceptual distance between two images.
        
        Returns:
            LPIPS distance (lower = more similar)
        """
        # Convert PIL to tensor
        tensor1 = self.lpips_transform(img1).unsqueeze(0).to(self.device)
        tensor2 = self.lpips_transform(img2).unsqueeze(0).to(self.device)
        
        # LPIPS expects images in [-1, 1] range
        tensor1 = tensor1 * 2.0 - 1.0
        tensor2 = tensor2 * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_loss = self.lpips_model(tensor1, tensor2)
        
        return lpips_loss.item()
    
    def _normalize_rewards(self, rewards: List[float]) -> List[float]:
        """
        Normalize rewards to have zero mean and unit variance.
        This stabilizes DDPO training.
        """
        rewards_array = np.array(rewards)
        
        # Use running statistics if available, otherwise use batch stats
        if len(self.reward_history) > 10:
            mean = np.mean(self.reward_history)
            std = np.std(self.reward_history)
        else:
            mean = np.mean(rewards_array)
            std = np.std(rewards_array)
        
        # Avoid division by zero
        if std < 1e-8:
            return rewards
        
        normalized = (rewards_array - mean) / (std + 1e-8)
        return normalized.tolist()
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics about recent rewards."""
        if not self.reward_history:
            return {}
        
        return {
            'mean_reward': float(np.mean(self.reward_history)),
            'std_reward': float(np.std(self.reward_history)),
            'min_reward': float(np.min(self.reward_history)),
            'max_reward': float(np.max(self.reward_history)),
        }