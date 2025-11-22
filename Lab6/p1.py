# image_segmentation_demo.py
# A single-file Python script to demonstrate and compare image segmentation methods:
# - Thresholding (global Otsu's and adaptive)
# - Region-based (simple region growing + watershed)
# - Edge-based (Canny edge detection with contour extraction)
# - Semantic segmentation (pre-trained DeepLabV3 ResNet50 from PyTorch/Torchvision)
#
# This script follows the exact procedure you provided, with detailed implementation,
# step-by-step execution, visualizations, and analysis. It uses only built-in sample images
# from scikit-image (no downloads required). Run this file on your computer with the required
# libraries installed (pip install scikit-image matplotlib torch torchvision).
#
# Real sources:
# - scikit-image: https://scikit-image.org/ (for thresholding, watershed, Canny, etc.)
# - Otsu's method: Nobuyuki Otsu, "A Threshold Selection Method from Gray-Level Histograms", 1979
# - Watershed: Luc Vincent and Pierre Soille, 1991
# - Canny edge detector: John Canny, 1986
# - DeepLabV3: Chen et al., "DeepLab: Semantic Image Segmentation...", 2017
# - Torchvision pre-trained models: https://pytorch.org/vision/stable/models.html
#
# Required libraries (install once):
# pip install scikit-image matplotlib torch torchvision numpy

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, filters, segmentation, feature, morphology
from skimage.measure import find_contours, label
from skimage.morphology import disk, binary_closing
from skimage.segmentation import watershed
from skimage.transform import resize
from scipy import ndimage as ndi
import torch
from torchvision import models, transforms
from skimage.color import label2rgb

# ----------------------------- Helper Functions -----------------------------
def simple_region_growing(gray_img, seed_point, tolerance=15):
    """Simple region growing implementation (fixed tolerance from seed intensity).
    Step-by-step:
    1. Normalize image to 0-255 uint8.
    2. Start from seed, grow if neighbor intensity difference <= tolerance.
    3. Use 8-connectivity.
    """
    h, w = gray_img.shape
    gray_uint8 = img_as_ubyte(gray_img)  # Ensure uint8
    seed_value = gray_uint8[seed_point[0], seed_point[1]]
    segmented = np.zeros((h, w), dtype=bool)
    to_process = [(seed_point[0], seed_point[1])]
    
    while to_process:
        x, y = to_process.pop()
        if segmented[x, y]:
            continue
        if abs(int(gray_uint8[x, y]) - int(seed_value)) <= tolerance:
            segmented[x, y] = True
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not segmented[nx, ny]:
                        to_process.append((nx, ny))
    return segmented

# Import this after skimage (safe)
from skimage.util import img_as_ubyte

# Load sample images (built-in, no internet needed)
print("Loading sample images...")
coins = data.coins()                     # Grayscale, simple high-contrast objects
astronaut = data.astronaut()             # Color, medium complexity (person + background)
chelsea = data.chelsea()                 # Color, natural scene (cat)

images = {"Coins (simple grayscale)": coins,
          "Astronaut (medium complexity)": astronaut,
          "Chelsea (cat - natural scene)": chelsea}

# PASCAL VOC class names (DeepLabV3 ResNet50 uses 21 classes)
voc_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
               'sofa', 'train', 'tvmonitor']

# Load pre-trained DeepLabV3 model once (takes a few seconds)
print("Loading pre-trained DeepLabV3 ResNet50 model...")
model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()

# Transform for semantic segmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------- 1. Thresholding Segmentation -----------------------------
print("\n1. Thresholding Segmentation")
for name, img in images.items():
    gray = color.rgb2gray(img) if img.ndim == 3 else img
    
    # Global Otsu's thresholding
    thresh_otsu = filters.threshold_otsu(gray)
    binary_otsu = gray > thresh_otsu
    
    # Adaptive thresholding
    thresh_adaptive = filters.threshold_local(gray, block_size=35, offset=10)
    binary_adaptive = gray > thresh_adaptive
    
    # Visualize
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(gray, cmap='gray')
    ax[0].set_title(f'Original - {name}')
    ax[1].imshow(binary_otsu, cmap='gray')
    ax[1].set_title("Otsu's Global Threshold")
    ax[2].imshow(binary_adaptive, cmap='gray')
    ax[2].set_title("Adaptive Threshold")
    for a in ax:
        a.axis('off')
    plt.suptitle("Thresholding Results")
    plt.show()

print("Discussion: Thresholding is fast and works well on high-contrast images (e.g., coins). "
      "It fails on non-uniform lighting or complex textures (see astronaut/cat).")

# ----------------------------- 2. Region-Based Segmentation -----------------------------
print("\n2. Region-Based Segmentation")
# Example on coins (classic watershed use case)
gray_coins = coins
# Markers for watershed
distance = ndi.distance_transform_edt(gray_coins > filters.threshold_otsu(gray_coins))
coords = feature.peak_local_max(distance, labels=gray_coins > filters.threshold_otsu(gray_coins), min_distance=20)
markers = np.zeros(distance.shape, dtype=bool)
markers[tuple(coords.T)] = True
markers = label(markers)
watershed_seg = watershed(-distance, markers, mask=gray_coins > filters.threshold_otsu(gray_coins))

# Region growing on coins (seed chosen manually for a coin)
seed_coin = (80, 200)  # Adjust if needed
region_grown = simple_region_growing(gray_coins, seed_coin, tolerance=20)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(gray_coins, cmap='gray')
ax[0].set_title('Original Coins')
ax[1].imshow(region_grown, cmap='gray')
ax[1].set_title('Region Growing (single region)')
ax[2].imshow(watershed_seg, cmap='nipy_spectral')
ax[2].set_title('Watershed (multiple regions)')
for a in ax:
    a.axis('off')
plt.suptitle("Region-Based Results on Coins")
plt.show()

print("Discussion: Region growing needs good seed points; poor choice leads to under-segmentation. "
      "Watershed separates touching objects well but can over-segment with noise.")

# ----------------------------- 3. Edge-Based Segmentation -----------------------------
print("\n3. Edge-Based Segmentation")
for name, img in images.items():
    gray = color.rgb2gray(img) if img.ndim == 3 else img
    edges = feature.canny(gray, sigma=2)
    # Close small gaps in edges
    edges_closed = binary_closing(edges, disk(3))
    contours = find_contours(edges_closed, 0.8)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(gray, cmap='gray')
    ax[0].set_title(f'Original - {name}')
    ax[1].imshow(edges_closed, cmap='gray')
    for contour in contours:
        ax[1].plot(contour[:, 1], contour[:, 0], linewidth=1, c='red')
    ax[1].set_title('Canny Edges + Contours')
    for a in ax:
        a.axis('off')
    plt.suptitle("Edge-Based Results")
    plt.show()

print("Discussion: Edges are clear on high-contrast areas but sensitive to noise and texture. "
      "Closed contours can define objects, but broken edges cause leakage.")

# ----------------------------- 4. Semantic Segmentation -----------------------------
print("\n4. Semantic Segmentation")
for name, img in [("Astronaut", astronaut), ("Chelsea (cat)", chelsea)]:
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)['out']
    
    pred = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()
    pred = resize(pred, img.shape[:2], order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)
    segmented_overlay = label2rgb(pred, image=img, alpha=0.4, bg_label=0)
    
    # Detected classes
    unique_labels = np.unique(pred)
    detected = [voc_classes[i] for i in unique_labels if i < len(voc_classes)]
    print(f"Detected classes in {name}: {detected}")
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title(f'Original - {name}')
    ax[1].imshow(segmented_overlay)
    ax[1].set_title('Semantic Segmentation (DeepLabV3)')
    for a in ax:
        a.axis('off')
    plt.show()

print("Discussion: Semantic segmentation understands scene context and labels classes correctly "
      "(e.g., 'person' in astronaut, 'cat' in chelsea). It handles complex natural scenes best.")

# ----------------------------- Analysis -----------------------------
print("\n=== Final Comparison and Reflection ===")
print("- Thresholding: Fastest, ideal for simple binary tasks (coins perfect). Fails on illumination variation.")
print("- Region-based: Good for touching objects (watershed on coins). Seed selection is critical.")
print("- Edge-based: Highlights boundaries well but noisy and rarely gives closed regions without post-processing.")
print("- Semantic: Most powerful for natural scenes (astronaut/cat). Understands object classes.")
print("- Effectiveness order on complex images: Semantic > Region-based > Edge-based > Thresholding")
print("- Computational complexity: Thresholding (instant) < Edge/Region (seconds) < Semantic (few seconds on CPU)")
print("- Limitations: Classic methods struggle with textures/noise; deep learning needs GPU for speed and large training data.")
print("- Use cases: Thresholding (document scanning), Watershed (microscopy), Edge (industrial inspection), Semantic (autonomous driving, medical imaging).")
