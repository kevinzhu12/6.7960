import torch
from transformers import pipeline
from PIL import Image
from clip_classifier import load_cod10k_lazy

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

print("Loading dataset...")
dataset = load_cod10k_lazy()

label_names = dataset['train'].features['label'].names
print(f"Found {len(label_names)} labels: {label_names[:10]}")

labels = [f"a photo of a {label}" for label in label_names]

clip = pipeline(
    task="zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    dtype=torch.bfloat16,
    use_fast=True,
    device=0
)

img_name = dataset['train'][0]['image']
# source_img = Image.open(img_name)
# sorted by score list of dict [{'score': float, 'label': str}, ...] for all labels
likelihoods = clip(img_name, labels) 

print("\nTop 5 predictions:")
for i, result in enumerate(likelihoods[:5]):
    print(f"{i+1}. '{result['label']}' with confidence: {result['score']:.2f}")

true_label_idx = dataset['train'][0]['label']
true_label = label_names[true_label_idx]
print(f"True label: {true_label}")
