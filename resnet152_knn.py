import cv2
import numpy as np
from tqdm import tqdm
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import pickle
import matplotlib.pyplot as plt

features_list = []
path_list = os.listdir('dataset')

try:
    model_ONNX = os.path.join('models', 'resnet152-v1-7.onnx')
    model = cv2.dnn.readNetFromONNX(model_ONNX)

    model_knn = os.path.join('models', 'knn_model.pkl')

    with open(model_knn, 'rb') as f:
        knn = pickle.load(f)
except Exception as e:
    print(e)

def preprocess():
    
    for file_path in tqdm(path_list, desc="Loading images"):
        file_path = os.path.join('dataset',file_path)
        features = extract_features(file_path)
        if not features is None:
            features_list.append(features.tolist())

def make_model():
    preprocess()
    train = normalize(np.array(features_list))
    np.savetxt(os.path.join('models', 'train.txt'), train, delimiter=',')
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')  # or 'euclidean', 'manhattan', etc.
    knn.fit(train)
    with open(model_knn, 'wb') as f:
        pickle.dump(knn, f)
        

def extract_color_features(img, bins=32):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Color Histograms
    hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256]).flatten()
    hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256]).flatten()

    # Color Moments
    mean_r, std_r = cv2.meanStdDev(img[:, :, 0])
    mean_g, std_g = cv2.meanStdDev(img[:, :, 1])
    mean_b, std_b = cv2.meanStdDev(img[:, :, 2])

    moments = np.array([mean_r, std_r, mean_g, std_g, mean_b, std_b]).flatten()

    # Normalize histograms and combine features
    hist_features = np.concatenate([hist_r, hist_g, hist_b])
    return np.concatenate([hist_features, moments])

# Preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=1.0 / 255, size=(224, 224), mean=(0.485, 0.456, 0.406), swapRB=True, crop=False
    )
    return blob

# Run inference
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        input_blob = preprocess_image(img)
        model.setInput(input_blob)
        shape_features = model.forward().flatten()
        color_features = extract_color_features(img)
        combined_features = np.concatenate([shape_features, color_features])
        return combined_features
    except Exception as e:
        print(str(e))
        return None

#  Show most similar images in order
def get_similar_products(image_path):
    query_feature = normalize(extract_features(image_path).reshape(1, -1))
    np.savetxt(os.path.join('models', 'query.txt'), query_feature, delimiter=',')
    distances, indices = knn.kneighbors(query_feature)
    results = indices[0][:5].tolist()
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 5))
    counter = 0

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    axes[counter].imshow(image_rgb)
    axes[counter].set_title(image_path)
    axes[counter].axis('off')
    counter += 1

    for result in results:
        file_name = path_list[result]
        fp = os.path.join('dataset', file_name)

        image = cv2.imread(fp)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        axes[counter].imshow(image_rgb)
        axes[counter].set_title(file_name)
        axes[counter].axis('off')
        counter += 1

        fig.savefig(os.path.join('output',f'o_resnet152_knn.png'))


# To make the model, you can change the dataset and run this code below
# make_model()

get_similar_products('new.webp')