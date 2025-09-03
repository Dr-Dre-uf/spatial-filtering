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

# --- Image Normalization and Augmentation ---
st.header("Image Normalization and Augmentation")
st.write("""
In this section, we apply **normalization** and **augmentation** techniques to both **Fluorescence (IF)** and **Brightfield (BF)** images.
1.  **Normalization**: Normalization scales the pixel values of the images to the range [0, 1], ensuring that all input images have consistent value ranges. This helps improve model training stability and speeds up convergence.
2.  **Augmentation**: Augmentation introduces random transformations like flipping and rotation to the images. This increases the diversity of the dataset and helps the model generalize better, thus preventing overfitting by simulating different real-world conditions.
""")

# Fluorescence Images
st.subheader("Fluorescence Images")
normalization_factor_if = st.slider("Normalization Factor (IF)", 1.0, 255.0, 255.0)
flip_horizontal_if = st.checkbox("Flip Horizontal (IF)")
flip_vertical_if = st.checkbox("Flip Vertical (IF)")
rotation_angle_if = st.slider("Rotation Angle (IF)", -30.0, 30.0, 0.0)
col1, col2 = st.columns(2)
with col1:
    st.image(image_if, caption="Original IF", use_container_width=True)
with col2:
    image_if_normalized = normalize_image(image_if, normalization_factor_if)
    image_if_augmented = augment_image(image_if_normalized, flip_horizontal_if, flip_vertical_if, rotation_angle_if)
    st.image(image_if_augmented, caption="Augmented IF", use_container_width=True)

# Brightfield Images
st.subheader("Brightfield Images")
normalization_factor_bf = st.slider("Normalization Factor (BF)", 1.0, 255.0, 255.0)
flip_horizontal_bf = st.checkbox("Flip Horizontal (BF)")
flip_vertical_bf = st.checkbox("Flip Vertical (BF)")
rotation_angle_bf = st.slider("Rotation Angle (BF)", -30.0, 30.0, 0.0)
col3, col4 = st.columns(2)
with col3:
    st.image(image_bf, caption="Original BF", use_container_width=True)
with col4:
    image_bf_normalized = normalize_image(image_bf, normalization_factor_bf)
    image_bf_augmented = augment_image(image_bf_normalized, flip_horizontal_bf, flip_vertical_bf, rotation_angle_bf)
    st.image(image_bf_augmented, caption="Augmented BF", use_container_width=True)

# --- Basic Filtering ---
st.header("Basic Image Filtering")
st.subheader("Grayscale Conversion")
grayscale_if = st.checkbox("Convert to Grayscale (IF)")
grayscale_bf = st.checkbox("Convert to Grayscale (BF)")
col5, col6 = st.columns(2)
with col5:
    st.image(image_if, caption="Original IF", use_container_width=True)
    image_if_gray = ImageOps.grayscale(image_if) if grayscale_if else image_if
    st.image(image_if_gray, caption="Gray IF", use_container_width=True)
with col6:
    st.image(image_bf, caption="Original BF", use_container_width=True)
    image_bf_gray = ImageOps.grayscale(image_bf) if grayscale_bf else image_bf
    st.image(image_bf_gray, caption="Gray BF", use_container_width=True)

# --- Edge Detection Filters ---
st.header("Edge Detection")
st.write("""
**Horizental/Vertical and Sobel Filters for Edge Detection**

*   Apply **horizontal and vertical edge filters** to detect directional features.

*   Use the **Sobel operator** to enhance edges and structural details.

*   Compare the results of different edge detection techniques.

By analyzing edge features, we can enhance image contrast and extract critical information for further medical image analysis. Let's begin!
""")

st.subheader("Horizontal Edge Detection")
horiz_strength_if = st.slider("Horizontal Edge Strength (IF)", 0.0, 1.0, 0.5)
IF_horiz = apply_horizontal_edge(np.array(image_if_gray), horiz_strength_if)

st.subheader("Vertical Edge Detection")
vert_strength_if = st.slider("Vertical Edge Strength (IF)", 0.0, 1.0, 0.5)
IF_vert = apply_vertical_edge(np.array(image_if_gray), vert_strength_if)

st.subheader("Sobel Edge Detection")
sobel_strength_if = st.slider("Sobel Edge Strength (IF)", 0.0, 1.0, 0.5)
IF_sobel = apply_sobel_edge(np.array(image_if_gray), sobel_strength_if)

col7, col8, col9 = st.columns(3)
with col7:
    st.image(image_if_gray, caption="Original IF", use_container_width=True)
with col8:
    st.image(IF_horiz, caption="Horizontal Edges IF", use_container_width=True)
    st.image(IF_vert, caption="Vertical Edges IF", use_container_width=True)
with col9:
    st.image(IF_sobel, caption="Sobel Edges IF", use_container_width=True)

# --- Motion Blur ---
st.header("Motion Blur")
motion_length_if = st.slider("Motion Blur Length (IF)", 5, 50, 20)
motion_angle_if = st.slider("Motion Blur Angle (IF)", 0.0, 180.0, 45.0)

motion_kernel = motion_blur_kernel(motion_length_if, motion_angle_if)
IF_motion = cv2.filter2D(np.array(image_if_gray), -1, motion_kernel)

col10, col11 = st.columns(2)
with col10:
    st.image(image_if_gray, caption="Original IF", use_container_width=True)
with col11:
    st.image(IF_motion, caption="Motion Blur IF", use_container_width=True)
    st.header("Adding Salt-and-Pepper Noise and Applying Denoising Filters")
st.write("""
In this section, we add salt-and-pepper noise to the images to simulate noisy conditions that commonly occur in real-world imaging scenarios. Salt-and-pepper noise is characterized by randomly occurring white and black pixels. After introducing the noise, we apply various denoising filters to remove the noise and recover the original image quality. The filters used in this process include median filtering and Gaussian smoothing, which are effective in reducing noise while preserving important image details.
""")

# Salt & Pepper Noise
snp_amount_if = st.slider("Salt & Pepper Noise Amount (IF)", 0.0, 0.1, 0.05)
snp_amount_bf = st.slider("Salt & Pepper Noise Amount (BF)", 0.0, 0.1, 0.05)

IF_snp = random_noise(np.array(image_if_gray), mode="s&p", amount=snp_amount_if)
BF_snp = random_noise(np.array(image_bf_gray), mode="s&p", amount=snp_amount_bf)

# Display noisy images
col12, col13 = st.columns(2)
with col12:
    st.image(image_if_gray, caption="Original IF", use_container_width=True)
    st.image(IF_snp, caption="Noisy IF", use_container_width=True)
with col13:
    st.image(image_bf_gray, caption="Original BF", use_container_width=True)
    st.image(BF_snp, caption="Noisy BF", use_container_width=True)