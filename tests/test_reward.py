"""Test the reward function with a small sample"""
import torch
from PIL import Image
from reward import CamouflageRewardFunction
from dataset import load_dataset
import os


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

def test_reward_function():
    print("=" * 50)
    print("Testing Reward Function")
    print("=" * 50)
    
    # Load a small sample from dataset
    dataset_root = "datasets/COD10K-v3"
    enhanced_root = "datasets/COD10K-v3-baseSD"
    dataset = load_dataset(dataset_root)
    train_dataset = dataset['train']
    
    # Get just 2 samples for testing
    print(f"\nDataset has {len(train_dataset)} samples")
    print("Taking first 2 samples for testing...")
    
    # Get label names
    label_names = train_dataset.features['label'].names
    labels = [f"a photo of a {label}" for label in label_names]
    
    # Get list of filenames from original dataset (sorted, matching dataset order)
    original_image_dir = os.path.join(dataset_root, "Train", "Image")
    original_filenames = sorted([f for f in os.listdir(original_image_dir) if f.endswith(".jpg")])
    
    # Initialize reward function
    print("\nInitializing reward function...")
    reward_fn = CamouflageRewardFunction(
        clip_model_name="openai/clip-vit-base-patch32",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_lpips_penalty=True,
        lpips_weight=0.1,
        normalize_rewards=False,  # Disable for testing to see raw values
    )
    
    # Test with 2 samples
    test_samples = []
    for i in range(min(2, len(train_dataset))):
        sample = train_dataset[i]
        
        # Load original image (already decoded by dataset)
        original_img = sample['image'].convert("RGB")
        
        # Get filename by index (dataset loads files in sorted order)
        filename = original_filenames[i]
        enhanced_path = os.path.join(enhanced_root, "Train", "Image", filename)
        
        # Resize original image
        from dataset import resize_with_crop
        original_img = resize_with_crop(original_img, (512, 512))
        
        # Load enhanced image
        if os.path.exists(enhanced_path):
            from diffusers.utils import load_image
            enhanced_img = load_image(enhanced_path).convert("RGB")
            enhanced_img = resize_with_crop(enhanced_img, (512, 512))
        else:
            # Fallback: use original if enhanced not found
            print(f"Warning: Enhanced image not found at {enhanced_path}, using original")
            enhanced_img = original_img
        
        # Get label
        true_label_idx = sample['label']
        true_label_name = label_names[true_label_idx]
        true_label_text = f"a photo of a {true_label_name}"
        
        test_samples.append({
            'image': enhanced_img,  # Use enhanced image
            'metadata': {
                'original_image': original_img,  # Use original image
                'true_label': true_label_text,
                'label_names': labels,
            }
        })
    
    # Test reward computation
    print("\nComputing rewards...")
    images = [s['image'] for s in test_samples]
    prompts = [''] * len(test_samples)
    metadata = [s['metadata'] for s in test_samples]

    
    rewards = reward_fn(images, prompts, metadata)
    
    print(f"\n✓ Reward function works!")
    print(f"  Sample 1 reward: {rewards[0]:.4f}")
    print(f"  Sample 2 reward: {rewards[1]:.4f}")
    
    # Test stats
    stats = reward_fn.get_stats()
    print(f"\n✓ Stats function works!")
    print(f"  {stats}")
    
    return True

if __name__ == "__main__":
    try:
        test_reward_function()
        print("\n" + "=" * 50)
        print("✓ All reward function tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()