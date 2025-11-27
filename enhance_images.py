import torch
import os
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
from tqdm import tqdm

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

device = "cuda"
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

print("Loading Stable Diffusion img2img pipeline...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to(device)

prompt = ''  # Empty prompt for enhancement
strength = 0.3
guidance_scale = 3

def resize_with_crop(img, target_size):
    """Resize image preserving aspect ratio, then crop to target size."""
    # Calculate scaling factor to ensure image covers target size
    target_w, target_h = target_size
    img_w, img_h = img.size
    
    # Scale so the smaller dimension matches target, larger dimension will be >= target
    scale = max(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    # Resize maintaining aspect ratio
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Crop center to target size
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    
    return img.crop((left, top, right, bottom))

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, "datasets", "COD10K-v3")
ENHANCED_ROOT = os.path.join(SCRIPT_DIR, "datasets", "COD10K-v3-enhanced")

# Create output directories
for split in ['Train', 'Test']:
    split_dir = os.path.join(ENHANCED_ROOT, split, "Image")
    os.makedirs(split_dir, exist_ok=True)

# Process each split directly from file system to preserve filenames
for split in ['Train', 'Test']:
# for split in ['Train']:
    split_path = os.path.join(DATASET_ROOT, split, "Image")
    
    if not os.path.isdir(split_path):
        print(f"Warning: Directory not found: {split_path}")
        continue
    
    # Get all image files in sorted order (matching dataset order)
    image_files = [f for f in sorted(os.listdir(split_path)) if f.endswith(".jpg")]
    num_samples = len(image_files)
    
    print(f"\nEnhancing {split} set ({num_samples} images)...")
    
    output_dir = os.path.join(ENHANCED_ROOT, split, "Image")
    
    for filename in tqdm(image_files):
        input_path = os.path.join(split_path, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Skip if already processed
        if os.path.exists(output_path):
            continue
        
        try:
            # Load and resize image with crop (preserves aspect ratio)
            init_image = load_image(input_path).convert("RGB")
            init_image = resize_with_crop(init_image, (512, 512))
            
            # Enhance with img2img
            enhanced_image = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale
            ).images[0]
            
            # Save enhanced image with original filename
            enhanced_image.save(output_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

print("\nEnhancement complete!")