import torch
import numpy as np
from collections import defaultdict
from transformers import pipeline
from PIL import Image
from clip_classifier import load_cod10k_lazy
import math
from tqdm import tqdm

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False


def eval(dataset_split, num_samples=None):

    if num_samples is None:
        num_samples = len(dataset_split)
    else:
        num_samples = min(num_samples, len(dataset_split))
    

    nlls = []
    scores = []
    correct = 0
    class_nlls = defaultdict(list)
    class_scores = defaultdict(list)
    epsilon = 1e-10

    print(f"Computing NLL for {num_samples} samples")

    for i in tqdm(range(num_samples)):

        if i % 100 == 0:
            print(f"Processing sample {i} of {num_samples}")
    
        sample = dataset_split[i]
        image = sample['image']
        true_label_idx = sample['label']
        true_label_name = label_names[true_label_idx]

        likelihoods = clip(image, labels)

        true_label_text = f"a photo of a {true_label_name}"

        # find the score for the true label
        for pred in likelihoods:
            if pred['label'] == true_label_text:
                score = pred['score']
                break

        scores.append(score)
        class_scores[true_label_name].append(score)

        nll = -math.log(score+epsilon)
        nlls.append(nll)
        class_nlls[true_label_name].append(nll)

        if true_label_text == likelihoods[0]['label']:
            correct += 1
    
    mean_nll = np.mean(nlls)
    std_nll = np.std(nlls)
    mean_probability = np.mean(scores)
    std_probability = np.std(scores)
    accuracy = correct / num_samples
    class_mean_nlls = {k: np.mean(v) for k, v in class_nlls.items()}
    class_mean_scores = {k: np.mean(v) for k, v in class_scores.items()}

    print(f"NLL results (n={num_samples}):")
    print(f"NLL: {mean_nll:.4f} ± {std_nll:.4f}")
    print(f"Probability: {mean_probability:.4f} ± {std_probability:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return {
        'mean_nll': mean_nll,
        'std_nll': std_nll,
        'mean_probability': mean_probability,
        'std_probability': std_probability,
        'class_mean_nlls': class_mean_nlls,
        'class_mean_scores': class_mean_scores,
        'accuracy': accuracy,
        'correct': correct,
        'num_samples': num_samples
    }

if __name__ == "__main__":

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


    results = eval(dataset['train'], num_samples=None)
    print(results)
