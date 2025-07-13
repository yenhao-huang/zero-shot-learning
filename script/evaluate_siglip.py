import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from utils.data_utils import get_imagenet_classnames, load_imagenet_index
from PIL import Image

if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load ImageNet subset index
    index = load_imagenet_index("data/data", "labels.csv", subset_size=100)

    # Load SigLIP model and processor
    model_name = "google/siglip-base-patch16-224"
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    # Prepare class names and encode once
    classnames = get_imagenet_classnames()
    with torch.no_grad():
        text_inputs = processor(
            text=classnames, return_tensors="pt", padding="max_length"
        ).to(device)
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

    # Evaluation loop
    correct = 0

    for img_path, label in tqdm(index, desc="Evaluating"):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_embed = model.get_image_features(**inputs)  # [1, hidden_dim]
            image_embed = torch.nn.functional.normalize(image_embed, dim=-1)

        similarity = torch.matmul(image_embed, text_embeds.T)  # [1, num_classes]
        pred_idx = similarity.argmax(dim=1).item()

        correct += int(pred_idx == label)
        print(
            f"Pred: {classnames[pred_idx]} | GT: {classnames[label]} | {'✅' if pred_idx == label else '❌'}"
        )

    accuracy = correct / len(index)
    print(f"\nZero-shot Accuracy on {len(index)} samples: {accuracy:.2%}")
