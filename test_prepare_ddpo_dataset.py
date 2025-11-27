"""Minimal test"""
from dataset import load_dataset, prepare_ddpo_dataset

dataset = load_dataset("datasets/COD10K-v3")
train_dataset_raw = dataset['train']
label_names = train_dataset_raw.features['label'].names
labels = [f"a photo of a {label}" for label in label_names]

# Test with first 2 samples
test_subset = train_dataset_raw.select(range(2))
ddpo_dataset = prepare_ddpo_dataset(test_subset, labels)

print(f"Dataset size: {len(ddpo_dataset)}")
print(f"Sample keys: {ddpo_dataset[0].keys()}")
print(f"Metadata keys: {ddpo_dataset[0]['metadata'].keys()}")
print(f"Image size: {ddpo_dataset[0]['image'].size}")
print(f"True label: {ddpo_dataset[0]['metadata']['true_label']}")

import pdb; pdb.set_trace()