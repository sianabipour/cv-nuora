# CV-Nuora: Image-Based Item Search

CV-Nuora is a Python-based project designed to search for items on [Nuora Shop](https://www.nuorashop.ir) using image input. It employs multiple methods for feature extraction and similarity matching to deliver accurate and efficient search results.

## Features

- **SIFT-based Search**: Extracts features using the Scale-Invariant Feature Transform (SIFT) algorithm for image matching.
- **Deep Learning with ResNet152**: Leverages a pre-trained ResNet152 model to perform similarity search using deep feature embeddings.
- **Customizable Dataset**: Supports user-defined datasets for tailored search results.
- **Output Visualization**: Stores the top five similar images for each search in the `output/` folder as plotted images.

## File Overview

- **`matcher.py`**: Implements matching algorithms based on extracted features for image search.
- **`image_hash.py`**: Provides hashing-based techniques for quick and efficient image similarity checks.
- **`resnet152_knn.py`**: Uses deep features from ResNet152 for image matching via k-nearest neighbors.

## Folder Structure

- **`dataset/`**: Contains sample images used for testing. Replace with your dataset to customize results.
- **`models/`**: Stores the models used in the project (excluding ResNet152, which must be downloaded separately).
- **`output/`**: Stores the top five similar images for each input as a plot for easy visualization.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Required libraries: Install dependencies using the following command:
  ```bash
  pip install -r requirements.txt

### Download the ResNet152 Model

To use the ResNet152-based search, download a suitable ResNet152 model from the [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet/model). Save the downloaded `resnet152-v1-7.onnx` file in the `models` folder.

### Run the Application

1. Clone the repository:
   ```bash
   git clone https://github.com/sianabipour/cv-nuora.git
   cd cv-nuora
2. The `dataset/` folder contains images already in the database. Add or replace these images to customize the database. The `new.webp` file serves as the input image, and the main scripts will try to match this image with the dataset to identify the most similar ones.

3. Execute one of the main scripts to perform image search:
   - **SIFT-based search**:
     ```bash
     python matcher.py
     ```
   - **Hash-based search**:
     ```bash
     python image_hash.py
     ```
   - **ResNet152-based search**:
     ```bash
     python resnet152_knn.py
     ```

The top five similar images for each search will be stored in the `output/` folder as plots.

## Customization

- Replace the contents of the `dataset/` folder with your own images to personalize the database.
- Adjust the `new.webp` input image to test against different queries.
- Experiment with different models or modify the feature extraction logic in the scripts to suit your needs.

## Contributing

Feel free to fork the repository and submit pull requests for improvements or additional features. For significant changes, please open an issue to discuss your ideas.
