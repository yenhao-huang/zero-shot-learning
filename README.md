## How to Run

conda activate object_detection

python script/data_prep.py

python script/evaluate.py

## Dataset
| Item         | Description                                                                |
| ------------ | -------------------------------------------------------------------------- |
| **Name**     | ImageNet-1K (ILSVRC 2012) Validation Set                                   |
| **Source**   | [Hugging Face Datasets](https://huggingface.co/datasets/imagenet-1k)       |
| **Size**     | 50,000 images (50 images per class)                                      |
| **Format**   | RGB images (JPEG), \~256×256 resolution                                    |
| **Examples** | dog, zebra, airplane, teapot, joystick, etc. (commonly recognized objects) |

ImageNet 全部 2GB

Another testing set
Mobulan/CUB-200-2011

## Model
clip-ViT-B-32
https://huggingface.co/sentence-transformers/clip-ViT-B-32

* Learns from WebImageText, which includes 400 M large-scale image-text pairs
* maximize similarity between matching pairs

Key Idea: find which class prompt is most semantically similar to a given image
>Given an image of a cat, the model compares its embedding with text prompts like
"a photo of a cat", "a photo of a dog", "a photo of a car"...
and the similarity with "a photo of a cat" is expected to be the highest.


| Item                       | Value                                         |
| -------------------------- | --------------------------------------------- |
| Parameters                 | \~151 million                                 |
| Embedding Dimension        | 512                                           |
| Max Sequence Length (text) | 77 tokens                                     |
| Image Input Size           | 224 × 224 pixels                              |

SigLIP
https://huggingface.co/google/siglip-base-patch16-224

Intro
https://blog.ritwikraha.dev/choosing-between-siglip-and-clip-for-language-image-pretraining

## Results

CLIP: 前 100 張 60%
SIGLIP: 73%

## Bugs

    text_inputs = processor(
        text=classnames, return_tensors="pt", padding="max_length"
    ).to(device)
    若 padding 改成 true 會導致 positional embedding 對齊問題準確率瞬間降成 0%