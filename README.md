# Zero-Shot Learning with Vision-Text Models
## How to Run

```bash
conda activate object_detection
python script/data_prep.py
python script/evaluate.py
```

---

## Dataset

| Item         | Description                                                              |
| ------------ | ------------------------------------------------------------------------ |
| **Name**     | ImageNet-1K (ILSVRC 2012) Validation Set                                 |
| **Source**   | [Hugging Face: imagenet-1k](https://huggingface.co/datasets/imagenet-1k) |
| **Size**     | 50,000 images (50 images per class)                                      |
| **Format**   | RGB JPEG images, \~256×256 resolution                                    |
| **Examples** | dog, zebra, airplane, teapot, joystick, etc.                             |

> Full ImageNet size after extraction: \~2GB

### Another Testing Set

* **Dataset**: [Mobulan/CUB-200-2011](https://huggingface.co/datasets/Mobulan/CUB-200-2011)
* **Use case**: Alternative for fine-grained image classification tasks

---

## Model: CLIP (clip-ViT-B-32)

* **Source**: [clip-ViT-B-32 on Hugging Face](https://huggingface.co/sentence-transformers/clip-ViT-B-32)
* **Pretraining Data**: WebImageText (400M image-text pairs)
* **Objective**: Maximize similarity between matched image-text pairs


| Item                       | Value                         |
| -------------------------- | ----------------------------- |
| Parameters                 | ~151 million                  |
| Vision Model               | ViT-B/32                      |
| Text Model                 | Transformer (12-layer, GPT-like) |
| Embedding Dimension        | 512                           |
| Max Sequence Length (text) | 77 tokens                     |
| Image Input Size           | 224 × 224 pixels              |


### Key Idea

> Given an image of a cat, the model computes similarity with prompts such as:
>
> * "a photo of a cat"
> * "a photo of a dog"
> * "a photo of a car"
>
> The similarity with "a photo of a cat" is expected to be the highest.

---

## Model: SigLIP

* **Source**: [siglip-base-patch16-224 on Hugging Face](https://huggingface.co/google/siglip-base-patch16-224)


| Item                       | Value                         |
| -------------------------- | ----------------------------- |
| Parameters                 | ~203 million                  |
| Vision Model               | ViT-B/16                      |
| Text Model                 | BERT-base (12-layer Transformer) |
| Embedding Dimension        | 768                           |
| Max Sequence Length (text) | 77 tokens                     |
| Image Input Size           | 224 × 224 pixels              |

---

## Results

| Model  | Accuracy (Top-1) |
| ------ | -------------------------------- |
| CLIP   | 60%                              |
| SigLIP | 73%                              |

---

## Bug Note

Using the SigLIP processor:

```python
text_inputs = processor(
    text=classnames, return_tensors="pt", padding="max_length"
).to(device)
```

> ⚠️ **Do not use `padding=True`**, or the accuracy will drop to **0%** due to **position embedding mismatch** between variable-length inputs and pretrained fixed-length expectations.
