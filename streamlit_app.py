import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from skimage import color
from skimage.util import random_noise

# --- Function Definitions ---
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

# --- Load Images ---
try:
    image_if = Image.open("./asset/IFCells.jpg")
    image_bf = Image.open("./asset/BloodSmear.png")
except FileNotFoundError as e:
    st.error(f"Error loading images: {e}.  Make sure images are in the same directory.")
    st.stop()

# --- Main Application ---
st.title("Spatial Filtering App")

# --- Image Selection ---
st.header("Image Selection")
image_type = st.radio("Select Image Type:", ("Fluorescence (IF)", "Brightfield (BF)"))

if image_type == "Fluorescence (IF)":
    image = image_if
    image_gray = image_if.convert("L")  # Keep original color for display, grayscale for processing
else:
    image = image_bf
    image_gray = image_bf.convert("L")  # Keep original color for display, grayscale for processing

# --- Noise Addition ---
st.header("Noise Addition")
snp_amount = st.slider("Salt & Pepper Noise Amount", 0.0, 0.1, 0.05)
noisy_image = random_noise(np.array(image_gray), mode="s&p", amount=snp_amount)

# --- Filtering ---
st.header("Filtering")

# Median Filtering
median_kernel_size = st.slider("Median Filter Kernel Size", 3, 15, 5)
if median_kernel_size % 2 == 0:
    median_kernel_size += 1
median_filtered_image = cv2.medianBlur(np.uint8(noisy_image), median_kernel_size)

# Edge Detection
st.subheader("Edge Detection")
horiz_strength = st.slider("Horizontal Edge Strength", 0.0, 1.0, 0.5)
vert_strength = st.slider("Vertical Edge Strength", 0.0, 1.0, 0.5)
sobel_strength = st.slider("Sobel Edge Strength", 0.0, 1.0, 0.5)

horiz_edge = apply_horizontal_edge(np.array(image_gray), horiz_strength)
vert_edge = apply_vertical_edge(np.array(image_gray), vert_strength)
sobel_edge = apply_sobel_edge(np.array(image_gray), sobel_strength)

# Motion Blur
st.subheader("Motion Blur")
motion_length = st.slider("Motion Blur Length", 5, 50, 20)
motion_angle = st.slider("Motion Blur Angle", 0.0, 180.0, 45.0)
motion_kernel = motion_blur_kernel(motion_length, motion_angle)
motion_blurred_image = cv2.filter2D(np.array(image_gray), -1, motion_kernel)

# --- Side-by-Side Comparison ---
st.header("Side-by-Side Comparison")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Original")
    st.image(image, caption="Original", use_container_width=True)

with col2:
    st.subheader("Noisy")
    st.image(noisy_image, caption="Noisy", use_container_width=True)

with col3:
    st.subheader("Filtered")
    st.image(median_filtered_image, caption="Median Filtered", use_container_width=True)

with col4:
    st.subheader("Edge/Blur")
    st.image(motion_blurred_image, caption="Motion Blurred", use_container_width=True)
    st.image(horiz_edge, caption="Horizontal Edges", use_container_width=True)
    st.image(vert_edge, caption="Vertical Edges", use_container_width=True)
    st.image(sobel_edge, caption="Sobel Edges", use_container_width=True)