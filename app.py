import streamlit as st
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np


from utils import *


def load_model(model_path):
    state_dict = torch.load(model_path)
    encoder = get_vgg_encoder()
    decoder = get_decoder()

    model = AdaINModel(encoder, decoder,AdaIN(.000001))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Undo ImageNet normalization for (B,C,H,W) or (C,H,W) tensors."""
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    
    if tensor.ndim == 3:
        mean = mean.squeeze(0)
        std = std.squeeze(0)
    
    return tensor * std + mean


def process_images(content_image, style_image, model):
    content_h, content_w = content_image.size[1], content_image.size[0]
    print(f"Using content shape: ({content_h}, {content_w})")

    # Define normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Convert both to tensors
    content_tensor = F.to_tensor(content_image)
    style_tensor = F.to_tensor(style_image)

    # Resize style to match content size (preserving aspect ratio)
    style_tensor = F.resize(style_tensor, size=[content_h, content_w], antialias=True)

    # Add batch dim + normalize
    content_tensor = normalize(content_tensor).unsqueeze(0)
    style_tensor = normalize(style_tensor).unsqueeze(0)

    with torch.no_grad():
        generated = model(content_tensor, style_tensor)
        y_gen = generated["x_gen"]
        y_gen = denormalize(y_gen)
        y_gen = y_gen.permute(0, 2, 3, 1).cpu().squeeze()
        y_gen = torch.clamp(y_gen, 0, 1)
        y_gen = (y_gen.numpy() * 255).astype(np.uint8)
    print('original shape', (content_h, content_w))
    print('y_gen shape',y_gen.shape[-2],y_gen.shape[-3])
    return y_gen


def main():

    st.title("Style Transfer with AdaIN")
    
    model_path = 'model.pt'
    
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File uploaders for content and style images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Content Image")
        content_file = st.file_uploader("Upload content image", type=['png', 'jpg', 'jpeg'])
        if content_file:
            content_image = st.image(content_file, caption="Content Image")
            content_img = Image.open(content_file).convert('RGB')
    with col2:
        st.subheader("Style Image")
        style_file = st.file_uploader("Upload style image", type=['png', 'jpg', 'jpeg'])
        if style_file:
            style_image = st.image(style_file, caption="Style Image")
            style_img = Image.open(style_file).convert('RGB')
    
    if st.button("Generate Stylized Image"):
        if content_file is None or style_file is None:
            st.warning("Please upload both content and style images.")
            return
        
        try:
            with st.spinner("Generating stylized image..."):
                # Process the images
                result_image = process_images(content_img, style_img, model)
                
                st.subheader("Result")
                st.image(result_image, caption="Stylized Image")
                
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()