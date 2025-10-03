import cv2
import numpy as np
import os
from math import pi, cos, sin

def process_harris(image_path):
    """
    Applies Harris corner detection, dilates the corners for better visibility,
    marks them in red on the original image, saves the result in a subfolder,
    and displays it.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Compute Harris corner response
    corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Dilate the corner response to make corners more visible
    corners = cv2.dilate(corners, None)

    # Mark corners on the original image where response is above threshold
    image[corners > 0.01 * corners.max()] = [0, 0, 255]  # Red color

    # Create results directory if it doesn't exist
    os.makedirs('results/harris', exist_ok=True)

    # Save the result
    save_path = 'results/harris/result.jpg'
    cv2.imwrite(save_path, image)
    print(f"Harris result saved to {save_path}")

    # Display the result
    # cv2.imshow('Harris Corners', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def process_sift(image_path):
    """
    Applies SIFT feature detection, marks keypoints in red on the original image,
    saves the result in a subfolder, and displays it.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints
    keypoints = sift.detect(gray, None)

    # Draw keypoints in red
    result = cv2.drawKeypoints(image, keypoints, None, color=(0, 0, 255))

    # Create results directory if it doesn't exist
    os.makedirs('results/sift', exist_ok=True)

    # Save the result
    save_path = 'results/sift/result.jpg'
    cv2.imwrite(save_path, result)
    print(f"SIFT result saved to {save_path}")

    # Display the result
    # cv2.imshow('SIFT Keypoints', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def process_orb(image_path):
    """
    Applies ORB feature detection, marks keypoints in red on the original image,
    saves the result in a subfolder, and displays it.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints
    keypoints = orb.detect(gray, None)

    # Draw keypoints in red
    result = cv2.drawKeypoints(image, keypoints, None, color=(0, 0, 255))

    # Create results directory if it doesn't exist
    os.makedirs('results/orb', exist_ok=True)

    # Save the result
    save_path = 'results/orb/result.jpg'
    cv2.imwrite(save_path, result)
    print(f"ORB result saved to {save_path}")

    # Display the result
    # cv2.imshow('ORB Keypoints', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def process_hog(image_path):
    """
    Computes HOG descriptor, visualizes the orientations as red lines on the image,
    saves the result in a subfolder, and displays it.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Parameters for HOG
    cell_size = 8
    block_size = 16
    block_stride = 8
    nbins = 9

    # Crop image to be compatible with HOG computation (multiple of block_stride)
    h, w = gray.shape
    h = (h - block_size) // block_stride * block_stride + block_size
    w = (w - block_size) // block_stride * block_stride + block_size
    gray = gray[:h, :w]
    vis = image[:h, :w, :].copy()  # Use color image for visualization

    # Create HOG descriptor with adjusted win_size
    hog = cv2.HOGDescriptor((w, h), (block_size, block_size), (block_stride, block_stride), (cell_size, cell_size), nbins)

    # Compute HOG features
    h_features = hog.compute(gray)

    # Calculate number of blocks
    ny = ((h - block_size) // block_stride) + 1
    nx = ((w - block_size) // block_stride) + 1

    # Reshape features to (ny, nx, 2, 2, nbins)
    features = h_features.reshape(ny, nx, 2, 2, nbins)

    # Visualize
    for i in range(ny):
        for j in range(nx):
            # Average histogram over the block's cells
            avg_hist = features[i, j].mean(axis=(0, 1))
            magnitude = avg_hist.max()
            if magnitude < 0.01:
                continue  # Skip weak gradients
            max_bin = np.argmax(avg_hist)
            angle = max_bin * (pi / nbins)
            center_x = j * block_stride + block_size // 2
            center_y = i * block_stride + block_size // 2
            length = 8  # Fixed length for visibility; can adjust
            dx = (length / 2) * cos(angle)
            dy = (length / 2) * sin(angle)
            start = (int(center_x - dx), int(center_y - dy))
            end = (int(center_x + dx), int(center_y + dy))
            cv2.line(vis, start, end, (0, 0, 255), 1)

    # Create results directory if it doesn't exist
    os.makedirs('results/hog', exist_ok=True)

    # Save the result
    save_path = 'results/hog/result.jpg'
    cv2.imwrite(save_path, vis)
    print(f"HOG result saved to {save_path}")

    # Display the result
    # cv2.imshow('HOG Visualization', vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "sample_image.jpg"  # Input image path

    # Process with each detector/descriptor
    process_harris(image_path)
    process_sift(image_path)
    process_orb(image_path)# May skip if contrib not installed
    process_hog(image_path)