import os
import csv
from datasets import load_dataset
from PIL import Image


def save_imagenet_dataset(n, save_dir="data/data", csv_name="labels.csv"):
    os.makedirs(save_dir, exist_ok=True)

    # Streaming load
    dataset = load_dataset(
        "benjamin-paine/imagenet-1k", split="validation", streaming=True
    )

    # Prepare iterator
    it = iter(dataset)

    # Save images and labels
    records = []
    for i in range(n):
        example = next(it)
        image: Image.Image = example["image"]
        label = example["label"]

        # Save image
        img_filename = f"img_{i:04d}.jpg"
        img_path = os.path.join(save_dir, img_filename)
        image.save(img_path)

        # Record label
        records.append((img_filename, label))

    # Save labels
    with open(os.path.join(save_dir, csv_name), mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(records)

    print(f"✅ Saved {n} images and labels to '{save_dir}/'")


if __name__ == "__main__":
    save_imagenet_dataset(n=50000)
