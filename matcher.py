import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (500, 500))
    if image.shape[2] == 4:
        grayscale_image = image[:, :, 0]  # Assuming the first channel is grayscale
        alpha_channel = image[:, :, 3]  # Extract the alpha channel
        
        # Create a mask where the alpha channel is 255 (fully opaque)
        mask = alpha_channel != 255
        
        # Set pixels in the grayscale image to a specific value where alpha is 255 (optional)
        grayscale_image[mask] = 0  # Or any other value you prefer, here it's set to 0 (black)
        image = grayscale_image
    else:
        image = grayscale_image = image[:, :, 0]

    return image

def preprocess(image):
    cv2.imshow('Image', image)
    cv2.waitKey(delay)

    # Apply Canny edge detection
    edges = cv2.Canny(image, threshold1=150, threshold2=200)

    cv2.imshow('Image', edges)
    cv2.waitKey(delay)

    # Define a kernel for the morphological operations
    kernel = np.ones((5, 5), np.uint8)

    dilated_image = cv2.dilate(edges, kernel, iterations=1)

    cv2.imshow('Image', dilated_image)
    cv2.waitKey(delay)

    # Find contours in the edge-detected image
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    # Create a mask for the foreground objects
    mask = np.zeros_like(image)

    # Fill the contours in the mask (this step highlights the objects)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Use the mask to set the background to white
    background_removed = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow('Image', background_removed)
    cv2.waitKey(delay)

    keypoints, descriptors = sift.detectAndCompute(background_removed, None)

    image_with_keypoints = cv2.drawKeypoints(background_removed, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Image', image_with_keypoints)
    cv2.waitKey(delay)

    return keypoints, descriptors, background_removed


def bfmatcher_normal(image, new, keypoints_new, descriptors_new, keypoints, descriptors):
    print('-------- bfmatcher_normal -------->', end=' ')
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_new, descriptors)

    matches = [m for m in matches if m.trainIdx < len(keypoints_new) and m.queryIdx < len(keypoints)]
    matches = sorted(matches, key=lambda x:x.distance)
    matches = matches[:20]
    avg = np.mean([match.distance for match in matches])
    print(f"Average: {avg}")

    matched_image = cv2.drawMatches(image, keypoints, new, keypoints_new, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Image', matched_image)
    cv2.waitKey(delay)

    return avg

def bfmatcher_knn_ratio(image, new, keypoints_new, descriptors_new, keypoints, descriptors, threshold):
    print(f'bfmatcher_knn_ratio {threshold} -------->', end=' ')
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors_new, descriptors, k=2)

    # Apply ratio test to keep only good matches
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    good_matches = [m for m in good_matches if m.trainIdx < len(keypoints_new) and m.queryIdx < len(keypoints)]

    avg = np.mean([match.distance for match in good_matches])
    print(f"Average: {avg}")

    matched_image = cv2.drawMatches(image, keypoints, new, keypoints_new, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Image', matched_image)
    cv2.waitKey(delay)

    return avg

def flann_knn_ratio(image, new, keypoints_new, descriptors_new, keypoints, descriptors, threshold):
    print(f'flann_knn {threshold} -------->', end=' ')

    # Set up the FLANN matcher
    # Define the index parameters for FLANN, where algorithm 1 means KDTree
    # (suitable for SIFT and other floating-point descriptors)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)  # Higher checks mean more accurate but slower

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform KNN matching with k=2
    matches = flann.knnMatch(descriptors_new, descriptors, k=2)

    # Apply ratio test as per Lowe's paper
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    good_matches = [m for m in good_matches if m.trainIdx < len(keypoints_new) and m.queryIdx < len(keypoints)]
    
    avg = np.mean([match.distance for match in good_matches])
    print(f"Average: {avg}")

    matched_image = cv2.drawMatches(image, keypoints, new, keypoints_new, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Image', matched_image)
    cv2.waitKey(delay)

    return avg


if __name__ == "__main__":
    delay = 0
    sift = cv2.SIFT_create()
    new_fp = 'new.webp'
    new = load_image(new_fp)
    keypoints_new, descriptors_new, new = preprocess(new)

    path_list = os.listdir('dataset')

    data = {}
    for path in path_list:
        print(f'------------ {path} ------------')

        image = load_image(os.path.join('dataset', path))
        keypoints, descriptors, image = preprocess(image)

        data[path] = {
            'bfmatcher_normal' : bfmatcher_normal(image, new, keypoints_new, descriptors_new, keypoints, descriptors),
            'bfmatcher_knn_ratio 90' : bfmatcher_knn_ratio(image, new, keypoints_new, descriptors_new, keypoints, descriptors, 0.90),
            'flann_knn_ratio 90' : flann_knn_ratio(image, new, keypoints_new, descriptors_new, keypoints, descriptors, 0.90),
        }
        print('\n')


    cv2.destroyAllWindows()

    df = pd.DataFrame(data)

    for index, row in df.iterrows():

        print('*' * 20)
        print(f"Row name: {index}")
        sorted_series_asc = list(row.sort_values().items())[:5]
        fig, axes = plt.subplots(1, len(sorted_series_asc) + 1, figsize=(15, 5))
        counter = 0
        
        image = cv2.imread(new_fp)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        axes[counter].imshow(image_rgb)
        axes[counter].set_title(new_fp)
        axes[counter].axis('off')
        counter += 1

        for name, value in sorted_series_asc:
            fp = os.path.join('dataset', name)
            image = cv2.imread(fp)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            axes[counter].imshow(image_rgb)
            if not pd.isna(value):
                axes[counter].set_title(f'{name}\n{round(value)}')
            else:
                axes[counter].set_title(f'{name}\nNaN')
            axes[counter].axis('off')
            counter += 1

        fig.savefig(os.path.join('output',f'o_{index}.png'))