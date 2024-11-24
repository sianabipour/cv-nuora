import cv2
import numpy as np
import imagehash
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

def preprocess(path, delay=0):
    """Load an image, apply edge detection, dilation, and background removal."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (500, 500))

    # Apply Canny edge detection
    edges = cv2.Canny(image, threshold1=150, threshold2=200)
    cv2.imshow('Edges', edges)
    cv2.waitKey(delay)

    # Define a kernel for the morphological operations
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(edges, kernel, iterations=1)
    cv2.imshow('Dilated Image', dilated_image)
    cv2.waitKey(delay)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Sort contours by area and take the largest one
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]

        # Create a mask for the largest contour
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Apply mask to remove the background
        background_removed = cv2.bitwise_and(image, image, mask=mask)
    else:
        background_removed = image  # In case no contours are found

    cv2.imshow('Background Removed', background_removed)
    cv2.waitKey(delay)
    return background_removed

def compute_image_hash(image_path):
    """Compute a hash for the processed image using perceptual hash (pHash)."""
    processed_image = preprocess(image_path)
    pil_image = Image.fromarray(processed_image)
    return imagehash.phash(pil_image)

def hamming_distance(hash1, hash2):
    """Compute the Hamming distance between two image hashes."""
    return hash1 - hash2

if __name__ == "__main__":
    new_fp = 'new.webp'
    new_image_hash = compute_image_hash(new_fp)

    path_list = os.listdir('dataset')
    data = {}

    for path in path_list:
        dataset_image_path = os.path.join('dataset', path)
        dataset_image_hash = compute_image_hash(dataset_image_path)

        # Calculate Hamming distance
        distance = hamming_distance(new_image_hash, dataset_image_hash)
        data[path] = {'Hamming Distance': distance}
        print(f"Image: {path} - Hamming Distance: {distance}")

    # Convert results to DataFrame and display top 5 most similar images
    df = pd.DataFrame(data).T
    sorted_df = df.sort_values(by="Hamming Distance")

    # Plot the query image and top 5 similar images
    fig, axes = plt.subplots(1, 6, figsize=(15, 5))

    # Show the original image in the first position
    new_image = cv2.imread(new_fp)
    new_image_rgb = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(new_image_rgb)
    axes[0].set_title('Query Image')
    axes[0].axis('off')

    for i, (index, row) in enumerate(sorted_df.head(5).iterrows()):
        similar_image_path = os.path.join('dataset', index)
        similar_image = cv2.imread(similar_image_path)
        similar_image_rgb = cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB)

        axes[i+1].imshow(similar_image_rgb)
        axes[i+1].set_title(f'{index}\nDistance: {int(row["Hamming Distance"])}')
        axes[i+1].axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join('output', f'o_image_hash.png'))