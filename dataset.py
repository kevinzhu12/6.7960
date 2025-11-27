from datasets import Dataset, DatasetDict, Features, Image as ImageFeature, ClassLabel, Value
import os
import datasets
import tempfile
from PIL import Image
from diffusers.utils import load_image

datasets.disable_caching()  # Disable caching temporarily

SPLITS = ["Train", "Test"]
LABEL_INDEX = 5

def image_label_generator(split_name: str, dataset_root: str):
    # Generators are memory efficient, lazy, and expected by datasets library
    """
    Generator function to yield (image, label_name) pairs.
    
    Note that images are stored as image_paths, and decoded by datasets.
    """
    print(f"GENERATOR CALLED for {split_name} from {dataset_root}")
    # 'Train/Images' and 'Test/Images' structure
    split_path = os.path.join(dataset_root, split_name, "Image")
    
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Directory not found: {split_path}")

    for filename in sorted(os.listdir(split_path)):    
        if filename.endswith(".jpg"): # only jpg - i checked
            file_path = os.path.join(split_path, filename)
            label_name = get_label_from_filename(filename)
            
            yield {
                "image": file_path,
                "label_name": label_name
            }
        else:
            print(f"Not an image (harmless, skipped): {filename}")

def get_label_from_filename(filename: str) -> str:
    """Extracts the SubClass label from the COD10K filename."""
    parts = filename.split('-')
    
    if len(parts) <= LABEL_INDEX:
        # Get next best label at idx 3 - the superclass label
        if len(parts) <= 3:
            raise Exception(f"Malformed file name (too short): {filename}")
        else: 
            return parts[3]
    
    # Extract the SubClass name (e.g., 'Snake')
    return parts[LABEL_INDEX]

def load_dataset(dataset_root: str):

    dataset_dict = {}
    
    # schema is info datasets need to load image on demand from filepath
    features_schema = Features({
        "image" : ImageFeature(),
        "label_name": Value('string')
    })
    
    # Use a temporary cache directory to avoid corrupted cache issues
    temp_cache = tempfile.mkdtemp(prefix="datasets_cache_")
    
    for split in SPLITS:
        print(f"Loading {split} split from {dataset_root}...")
        raw_dataset = Dataset.from_generator(
            image_label_generator,
            features=features_schema,
            gen_kwargs={"split_name": split, "dataset_root": dataset_root}, # this are kwargs it passes to image_label_generator
            cache_dir=temp_cache
        )
        dataset_dict[split.lower()] = raw_dataset
        
    raw_datasets = DatasetDict(dataset_dict)
    
    # convert str class names to ClassLabel features
    all_labels = set()
    for split in raw_datasets:
        all_labels.update(raw_datasets[split]["label_name"])
        
    label_list = sorted(list(all_labels))
    # mapping of all possible labels to ints (categorical)
    class_feature = ClassLabel(names=label_list) 
    
    def encode_labels(sample):
        sample['label'] = class_feature.str2int(sample['label_name'])
        return sample
    
    print("Encoding labels...")
    raw_datasets = raw_datasets.map(encode_labels)
    
    # imgs remain untouched, and are decoded from filepaths JIT
    final_dataset = raw_datasets.map(
        encode_labels, 
        remove_columns=['label_name']
    )
    
    final_dataset = final_dataset.cast_column("label", class_feature)
    
    return final_dataset
    

# for DDPO training
def resize_with_crop(img, target_size):
    """Resize image preserving aspect ratio, then crop to target size."""
    target_w, target_h = target_size
    img_w, img_h = img.size
    
    scale = max(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    
    return img.crop((left, top, right, bottom))

def prepare_ddpo_dataset(dataset, labels):
    """
    Convert your dataset to DDPO format.
    Each sample needs: prompt, image, metadata
    
    Args:
        dataset: HuggingFace dataset with 'image' and 'label' columns
        labels: List of label strings like ["a photo of a BatFish", ...]
    
    Returns:
        List of dicts with keys: 'prompt', 'image', 'metadata'
    """
    ddpo_samples = []
    
    for sample in dataset:
        # Load original image
        if isinstance(sample['image'], str):
            # If it's a file path, load it
            original_image = load_image(sample['image']).convert("RGB")
        else:
            # If it's already a PIL Image
            original_image = sample['image'].convert("RGB")
        
        # Resize to 512x512 (matching your enhance_images.py)
        original_image = resize_with_crop(original_image, (512, 512))
        
        # Get true label
        true_label_idx = sample['label']
        true_label_name = dataset.features['label'].names[true_label_idx]
        true_label_text = f"a photo of a {true_label_name}"
        
        ddpo_samples.append({
            'prompt': '',  # Empty prompt for enhancement
            'image': original_image,
            'metadata': {
                'original_image': original_image,
                'true_label': true_label_text,
                'label_names': labels,
            }
        })
    
    return ddpo_samples