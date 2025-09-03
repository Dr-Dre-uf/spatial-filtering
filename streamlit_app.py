import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from skimage import color
from skimage.util import random_noise

# --- Function Definitions --- (Keep these as they are)
def augment_image(image, flip_horizontal, flip_vertical, rotation_angle):
    # Random horizontal flip
    if flip_horizontal:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # Random vertical flip
    if flip_vertical:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    # Random rotation (limited) - Pillow's rotation is not as flexible
    image = image.rotate(rotation_angle)
    return image

def normalize_image(image, normalization_factor):
    img_array = np.array(image)
    normalized_array = img_array / normalization_factor
    return Image.fromarray((normalized_array * 255).astype(np.uint8))  # Convert back to PIL Image

def add_salt_pepper_noise(image, amount):
    img_array = np.array(image)
    num_pixels = img_array.size
    num_noisy_pixels = int(amount * num_pixels)
    for _ in range(num_noisy_pixels):
        x = np.random.randint(0, img_array.shape[0])
        y = np.random.randint(0, img_array.shape[1])
        img_array[x, y] = np.random.choice([0, 255])  # 0 for black, 255 for white
    return Image.fromarray(img_array.astype(np.uint8))

def median_filter(image, kernel_size):
    img_array = np.array(image)
    kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1  # Ensure kernel size is odd
    pad_width = [(kernel_size // 2, kernel_size // 2)] * img_array.ndim  # Dynamically adjust padding based on image dimensions
    padded_img = np.pad(img_array, pad_width, mode='reflect')
    filtered_img = np.zeros_like(img_array)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            window = padded_img[i:i + kernel_size, j:j + kernel_size].flatten()
            filtered_img[i, j] = np.median(window)
    return Image.fromarray(filtered_img.astype(np.uint8))

def motion_blur_kernel(length, angle):
    """Creates a motion blur kernel"""
    kernel = np.zeros((length, length))
    kernel[int((length - 1) / 2), :] = np.ones(length)
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    kernel /= length
    return kernel

@st.cache_data
def apply_horizontal_edge(img, strength):
    horiz_filter = np.array([[1, 1, 1],
                              [0, 0, 0],
                              [-1, -1, -1]], dtype=np.float32)
    filtered_img = cv2.filter2D(img, -1, horiz_filter) * strength
    filtered_img = np.clip(filtered_img, 0.0, 1.0)
    return filtered_img

@st.cache_data
def apply_vertical_edge(img, strength):
    vert_filter = np.array([[1, 0, -1],
                             [1, 0, -1],
                             [1, 0, -1]], dtype=np.float32)
    filtered_img = cv2.filter2D(img, -1, vert_filter) * strength
    filtered_img = np.clip(filtered_img, 0.0, 1.0)
    return filtered_img

@st.cache_data
def apply_sobel_edge(img, strength):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    sobel_combined = np.clip(sobel_combined * strength, 0.0, 1.0)
    return sobel_combined

# --- Main Application ---
st.title("Spatial Filtering App")

# --- Image Selection ---
image_type = st.radio("Select Image Type:", ("Fluorescence (IF)", "Brightfield (BF)", "Upload Image"))

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
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_gray = image.convert("L")
    else:
        image = None
        image_gray = None

if image is None:
    st.stop()

# --- Noise Addition ---
st.header("Noise Addition")
snp_amount = st.slider("Salt & Pepper Noise Amount", 0.0, 0.1, 0.05)
noisy_image = random_noise(np.array(image_gray), mode="s&p", amount=snp_amount)

# --- Filtering and Display --- (The rest of your filtering and display code goes here)
# ... (You can copy your existing filtering and display code here)
# Example (Median Filtering):
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Image")
    st.image(image, caption="Original", use_container_width=True)
with col2:
    st.subheader("Noisy Image")
    st.image(noisy_image, caption="Noisy", use_container_width=True)

st.header("Filtering")

# Median Filtering
st.subheader("Median Filtering")
median_kernel_size = st.slider("Median Filter Kernel Size", 3, 15, 5)
if median_kernel_size % 2 == 0:
    median_kernel_size += 1
median_filtered_image = cv2.medianBlur(np.uint8(noisy_image), median_kernel_size)
col3, col4 = st.columns(2)
with col3:
    st.subheader("Original Image")
    st.image(image, caption="Original", use_container_width=True)
with col4:
    st.subheader("Median Filtered Image")
    st.image(median_filtered_image, caption="Median Filtered", use_container_width=True)

# Edge Detection
st.subheader("Edge Detection")
horiz_strength = st.slider("Horizontal Edge Strength", 0.0, 1.0, 0.5)
vert_strength = st.slider("Vertical Edge Strength", 0.0, 1.0, 0.5)
sobel_strength = st.slider("Sobel Edge Strength", 0.0, 1.0, 0.5)
horiz_edge = apply_horizontal_edge(np.array(image_gray), horiz_strength)
vert_edge = apply_vertical_edge(np.array(image_gray), vert_strength)
sobel_edge = apply_sobel_edge(np.array(image_gray), sobel_strength)
col5, col6 = st.columns(2)
with col5:
    st.subheader("Original Image")
    st.image(image, caption="Original", use_container_width=True)
with col6:
    st.subheader("Edge Detection Images")
    st.image(horiz_edge, caption="Horizontal Edges", use_container_width=True)
    st.image(vert_edge, caption="Vertical Edges", use_container_width=True)
    st.image(sobel_edge, caption="Sobel Edges", use_container_width=True)

# Motion Blur
st.subheader("Motion Blur")
motion_length = st.slider("Motion Blur Length", 5, 50, 20)
motion_angle = st.slider("Motion Blur Angle", 0.0, 180.0, 45.0)
motion_kernel = motion_blur_kernel(motion_length, motion_angle)
motion_blurred_image = cv2.filter2D(np.array(image_gray), -1, motion_kernel)
col7, col8 = st.columns(2)
with col7:
    st.subheader("Original Image")
    st.image(image, caption="Original", use_container_width=True)
with col8:
    st.subheader("Motion Blurred Image")
    st.image(motion_blurred_image, caption="Motion Blurred", use_container_width=True)