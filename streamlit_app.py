import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import random

# --- Function Definitions ---

def augment_image(image):
    # Random horizontal flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Random vertical flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Random rotation (limited) - Pillow's rotation is not as flexible
    angle = random.uniform(-30, 30)
    image = image.rotate(angle)

    return image

# --- Load Images ---
try:
    image_if = Image.open("./asset/IFCells.jpg")
    image_bf = Image.open("./asset/BloodSmear.png")
except FileNotFoundError as e:
    st.error(f"Error loading images: {e}.  Make sure images are in the same directory.")
    st.stop()

# --- Image Normalization ---
def normalize_image(image):
    img_array = np.array(image)
    normalized_array = img_array / 255.0
    return Image.fromarray((normalized_array * 255).astype(np.uint8))  # Convert back to PIL Image

image_if_normalized = normalize_image(image_if)
image_bf_normalized = normalize_image(image_bf)

# --- Image Augmentation ---
image_if_augmented = augment_image(image_if_normalized)
image_bf_augmented = augment_image(image_bf_normalized)


# --- Streamlit App ---
st.title("Clinical Image Processing Exercise (No Matplotlib/OpenCV)")

st.header("Image Normalization and Augmentation")

st.subheader("Fluorescence Images")
st.image([image_if, image_if_normalized, image_if_augmented], caption=["Original", "Normalized", "Augmented"], use_container_width=True)

st.subheader("Brightfield Images")
st.image([image_bf, image_bf_normalized, image_bf_augmented], caption=["Original", "Normalized", "Augmented"], use_container_width=True)

# --- Basic Filtering (Example - Grayscale) ---
st.header("Basic Image Filtering")
image_if_gray = ImageOps.grayscale(image_if)
image_bf_gray = ImageOps.grayscale(image_bf)

st.subheader("Grayscale Conversion")
st.image([image_if, image_if_gray, image_bf, image_bf_gray], caption=["Original IF", "Gray IF", "Original BF", "Gray BF"], use_container_width=True)