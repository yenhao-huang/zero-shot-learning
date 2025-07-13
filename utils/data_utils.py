# prepare_data.py
from datasets import load_dataset
import torch
from torchvision import transforms
from PIL import Image
import os
import csv


def get_imagenet_classnames():
    classnames = []
    with open("data/imagenet_classname.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                classnames.append(row[1].strip())
    return classnames


def load_imagenet_index(
    image_dir="data/data", label_file="labels.csv", subset_size=None
):
    """Return a list of (filepath, label) tuples – NO images loaded."""
    index = []
    with open(os.path.join(image_dir, label_file), newline="") as f:
        for row in csv.DictReader(f):
            index.append((os.path.join(image_dir, row["filename"]), int(row["label"])))
            if subset_size and len(index) >= subset_size:
                break
    return index


if __name__ == "__main__":
    # Just preview
    prompts = get_imagenet_classnames()
    dataset = load_imagenet_dataset()
    print("Loaded", len(dataset), "samples")
    print("Example:", dataset[0])
    print("Prompt example:", prompts[0])
