import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = 'sample_image.jpg'  # Replace with your image path
img = cv2.imread(image_path)
if img is None:
    print("Error: Could not load image.")
    exit()

# Part 1: Histogram Equalization
# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate histogram of original grayscale image
hist_orig = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

# Apply histogram equalization
equalized_img = cv2.equalizeHist(gray_img)

# Calculate histogram of equalized image
hist_eq = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])

# Part 2: Contrast Adjustment
# Linear Contrast Adjustment (Contrast Stretching)
min_val, max_val = np.min(gray_img), np.max(gray_img)
contrast_stretched = np.clip((gray_img - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)

# Calculate histogram of contrast-stretched image
hist_stretch = cv2.calcHist([contrast_stretched], [0], None, [256], [0, 256])

# Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray_img)

# Calculate histogram of CLAHE image
hist_clahe = cv2.calcHist([clahe_img], [0], None, [256], [0, 256])

# Part 5: Visualization
# Display images
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.title('Original Image')
plt.imshow(gray_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title('Histogram Equalized')
plt.imshow(equalized_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title('Contrast Stretched')
plt.imshow(contrast_stretched, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title('CLAHE')
plt.imshow(clahe_img, cmap='gray')
plt.axis('off')

# Plot histograms
plt.subplot(2, 4, 5)
plt.title('Original Histogram')
plt.plot(hist_orig)
plt.xlim([0, 256])

plt.subplot(2, 4, 6)
plt.title('Equalized Histogram')
plt.plot(hist_eq)
plt.xlim([0, 256])

plt.subplot(2, 4, 7)
plt.title('Contrast Stretched Histogram')
plt.plot(hist_stretch)
plt.xlim([0, 256])

plt.subplot(2, 4, 8)
plt.title('CLAHE Histogram')
plt.plot(hist_clahe)
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

# Save results
cv2.imwrite('equalized_image.jpg', equalized_img)
cv2.imwrite('contrast_stretched_image.jpg', contrast_stretched)
cv2.imwrite('clahe_image.jpg', clahe_img)