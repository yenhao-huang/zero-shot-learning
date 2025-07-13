import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import csv
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from sentence_transformers import SentenceTransformer, util
from utils.data_utils import get_imagenet_classnames, load_imagenet_index


if __name__ == "__main__":
    # Load ImageNet index
    index = load_imagenet_index("data/data", "labels.csv", subset_size=100)

    # Load CLIP model from SentenceTransformer
    model = SentenceTransformer("clip-ViT-B-32")

    # Prepare text prompts and encode them
    classnames = get_imagenet_classnames()
    text_prompts = [f"a photo of a {name}" for name in classnames]
    text_embeddings = model.encode(text_prompts, convert_to_tensor=True)

    # Run evaluation loop
    correct = 0
    total = len(index)

    for i, (img_path, label) in enumerate(tqdm(index, desc="Evaluating")):
        image = Image.open(img_path).convert("RGB")

        image_embedding = model.encode(image, convert_to_tensor=True)
        similarity = util.cos_sim(image_embedding, text_embeddings)[0]
        pred = similarity.argmax().item()

        correct += int(pred == label)
        print(
            f"[{i+1}/{total}] Pred: {classnames[pred]} | GT: {classnames[label]} | {'✅' if pred == label else '❌'}"
        )

    # Print final accuracy
    accuracy = correct / total
    print(f"\nZero-shot Accuracy on {total} samples: {accuracy:.2%}")
