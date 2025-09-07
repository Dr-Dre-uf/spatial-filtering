import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage import color
from skimage.filters import median  # Using skimage for median filtering
from scipy import ndimage  # For alternative edge detection and motion blur

# --- Function Definitions ---
def add_salt_pepper_noise(image, amount):
    img_array = np.array(image)
    num_pixels = img_array.size
    num_noisy_pixels = int(amount * num_pixels)
    for _ in range(num_noisy_pixels):
        x = np.random.randint(0, img_array.shape[0])
        y = np.random.randint(0, img_array.shape[1])
        img_array[x, y] = np.random.choice([0, 255])  # 0 for black, 255 for white
    return Image.fromarray(img_array.astype(np.uint8))

# --- Main Application ---
st.title("Spatial Filtering App")

# --- Sidebar ---
st.sidebar.header("Filtering Parameters")
image_type = st.sidebar.radio("Select Image Type:", ("Fluorescence (IF)", "Brightfield (BF)", "Upload Image"))
snp_amount = st.sidebar.slider("Salt & Pepper Noise Amount", 0.0, 0.1, 0.05)
median_kernel_size = st.sidebar.slider("Median Filter Kernel Size", 3, 15, 5)
if median_kernel_size % 2 == 0:
    median_kernel_size += 1
edge_strength = st.sidebar.slider("Edge Detection Strength", 0.0, 2.0, 1.0)
motion_length = st.sidebar.slider("Motion Blur Length", 5, 50, 20)
motion_angle = st.sidebar.slider("Motion Blur Angle", 0.0, 180.0, 45.0)

# --- Image Selection ---
if image_type == "Fluorescence (IF)":
    try:
        image = Image.open("./asset/IFCells.jpg")
        image_gray = image.convert("L")
    except FileNotFoundError:
        st.error("IFCells.jpg not found.")
        image = None
        image_gray = None
elif image_type == "Brightfield (BF)":
    try:
        image = Image.open("./asset/BloodSmear.png")
        image_gray = image.convert("L")
    except FileNotFoundError:
        st.error("BloodSmear.png not found.")
        image = None
        image_gray = None
else:  # Upload Image
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_gray = image.convert("L")
    else:
        image = None
        image_gray = None

if image is None:
    st.stop()

# --- Noise Addition ---
noisy_image = add_salt_pepper_noise(image_gray, snp_amount)

# --- Filtering ---
median_filtered_image = median(noisy_image, footprint=np.ones((median_kernel_size, median_kernel_size)))
sobel_edge = ndimage.sobel(noisy_image)
sobel_edge = (sobel_edge * edge_strength).clip(0, 1)
motion_blurred_image = ndimage.gaussian_filter(noisy_image, sigma=motion_length)

# --- Display ---
st.header("Filtering Results")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Original Image")
    st.image(image_gray, caption="Original", use_container_width=True)
with col2:
    st.subheader("Median Filtered")
    st.image(median_filtered_image, caption="Median Filtered", use_container_width=True)
with col3:
    st.subheader("Processed Image")
    st.image(sobel_edge, caption="Sobel Edges", use_container_width=True)
