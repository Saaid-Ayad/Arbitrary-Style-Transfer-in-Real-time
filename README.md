# AdaIN - Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization

A PyTorch implementation of **Adaptive Instance Normalization (AdaIN)** for real-time arbitrary style transfer. This project implements the paper "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" and provides both training and inference capabilities with a user-friendly Streamlit interface.

## Overview

AdaIN is a style transfer technique that uses instance normalization to align the mean and variance of content and style feature maps. Unlike other style transfer methods, AdaIN enables fast, flexible style transfer while maintaining high quality results.

### Key Features
- **Fast Style Transfer**: Real-time inference using pre-trained models
- **Arbitrary Styles**: Transfer any style image to any content image
- **Web Interface**: Easy-to-use Streamlit application for style transfer
- **Trainable Model**: PyTorch Lightning implementation for training on custom datasets
- **GPU Optimized**: Supports CUDA acceleration and multi-GPU training

## Project Structure

```
ada/
├── app.py              # Streamlit web interface for style transfer
├── train_lit.py        # PyTorch Lightning training script
├── utils.py            # Core AdaIN modules and utilities
├── model.pt            # Pre-trained model weights
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support, optional)


# Install dependencies
pip install torch torchvision streamlit pytorch-lightning
```

### Quick Start with Web Interface

Run the Streamlit app for interactive style transfer:

```bash
streamlit run app.py
```

Then:
1. Upload a content image
2. Upload a style image
3. Click "Generate Stylized Image"

### Training

To train the model on your own dataset:

```bash
python train_lit.py \
    --batch_size 8 \
    --epochs 10 \
    --lr 1e-4 \
    --content_weight 1.0 \
    --style_weight 14.0 \
    --devices 2
```

**Training Arguments:**
- `--images_train_path`: Path to training content images (default: "image_train/")
- `--images_val_path`: Path to validation content images (default: "image_val/")
- `--images_test_path`: Path to test content images (default: "image_test/")
- `--styles_train_path`: Path to training style images (default: "style_path/")

- `--lr`: Learning rate (default: 1e-4)
- `--batch_size`: Batch size (default: 8)
- `--epochs`: Number of training epochs (default: 10)
- `--content_weight`: Weight for content loss (default: 1.0)
- `--style_weight`: Weight for style loss (default: 14.0)
- `--devices`: Number of GPUs to use (default: 2)
- `--val_interval`: Validation interval as fraction of epoch (default: 0.25)
- `--checkpoint`: Path to load model checkpoint (optional)
- `--new_checkpoint_name`: Output checkpoint filename (default: "adain_final.pt")

## Architecture

### AdaIN Module
The core AdaIN operation normalizes content features and applies style statistics:

```
AdaIN(content, style) = σ(style) * (content - μ(content)) / σ(content) + μ(style)
```

### Model Components

1. **Encoder**: VGG-19 (pre-trained on ImageNet, frozen during training)
   - Extracts feature representations
   - Used for both content and style features

2. **AdaIN Layer**: Adaptive instance normalization
   - Aligns content and style statistics
   - Creates the stylized feature representation

3. **Decoder**: Custom CNN
   - Reconstructs images from stylized features
   - Trained to map stylized features back to image space

### Loss Functions

- **Content Loss**: Perceptual loss between generated and content features
  ```
  L_c = ||F(g(t)) - F(t)||_2
  ```

- **Style Loss**: Gram matrix statistics at multiple layers
  ```
  L_s = Σ ||mean(F_i(g(t))) - mean(F_i(s))||_2 + ||std(F_i(g(t))) - std(F_i(s))||_2
  ```

- **Total Loss**: 
  ```
  L = λ_c * L_c + λ_s * L_s
  ```

## Model Details

- **Input Size**: 224×224 (resized with aspect ratio preservation)
- **Encoder Depth**: VGG-19 features (up to layer 21)
- **Decoder Layers**: 13 convolutional layers
- **Parameter Count**: ~1.6M trainable parameters (decoder only)

## Dataset Structure

For training, organize your datasets as:

```
image_train/
├── img1.jpg
├── img2.jpg
└── ...

style_path/
├── style1.jpg
├── style2.jpg
└── ...

test_path/
├── test1.jpg
├── test2.jpg
└── ...
```

## Examples

### Basic Inference

```python
import torch
from utils import *
from app import load_model, process_images
from PIL import Image

# Load model
model = load_model('model.pt')

# Load images
content = Image.open('content.jpg').convert('RGB')
style = Image.open('style.jpg').convert('RGB')

# Generate stylized image
result = process_images(content, style, model)
```

## License

This project is open source and available for research and educational purposes.
