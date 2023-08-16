import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from tensorflow.keras.applications.vgg16 import preprocess_input

# Step 2: Write function to create metadata of the image
def create_metadata(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Assuming only one face is detected
        face_roi = gray_image[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (224, 224))
        
        # Convert grayscale image to RGB
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2RGB)
        
        # Preprocess input for VGG16
        preprocessed_face = preprocess_input(rgb_face)
        
        vgg_model = VGG16(weights='imagenet', include_top=False)
        embeddings = vgg_model.predict(np.expand_dims(preprocessed_face, axis=0))
        metadata = {'embeddings': embeddings.flatten()}
    else:
        metadata = {'embeddings': None}
    
    return metadata

# Step 3: Write a loop to iterate through images and create metadata
data_dir = 'faces_dataset'  # Directory containing subdirectories of images
metadata_list = []
labels = []

for actor_folder in os.listdir(data_dir):
    actor_path = os.path.join(data_dir, actor_folder)
    for image_file in os.listdir(actor_path):
        image_path = os.path.join(actor_path, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            metadata = create_metadata(image)
            metadata_list.append(metadata)
            labels.append(actor_folder)  # Assign label based on actor folder name

# Step 4: Generate Embeddings vectors using VGG16
def generate_embeddings(image):
    # Load VGG16 model with pre-trained weights
    vgg_model = VGG16(weights='imagenet', include_top=False)
    x = tf.image.resize(image, (224, 224))
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    embeddings = vgg_model.predict(x)
    return embeddings

# Step 5: Build distance metrics
def calculate_distances(embeddings1, embeddings2, metric='cosine'):
    if metric == 'cosine':
        distances = cosine_distances(embeddings1, embeddings2)
    return distances

# Step 6: Use PCA for dimensionality reduction
def apply_pca(data, n_components=128):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

# Step 7: Build SVM classifier
X = np.array([metadata['embeddings'] for metadata in metadata_list])
y = np.array(labels)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, y_encoded)

# Step 8: Import and display test images
test_image_paths = ['Benedict Cumberbatch9.jpg' ,'Dwayne Johnson4.jpg' ]  # Provide actual paths
test_images = [cv2.imread(image_path) for image_path in test_image_paths]

# Step 9: Use the trained SVM model to predict faces in test images
for test_image in test_images:
    test_metadata = create_metadata(test_image)
    if test_metadata['embeddings'] is not None:
        test_metadata_reduced = apply_pca(test_metadata['embeddings'].reshape(1, -1))
        predicted_label_encoded = svm_classifier.predict(test_metadata_reduced)
        predicted_label = label_encoder.inverse_transform(predicted_label_encoded)
        print(f"Predicted label for test image: {predicted_label}")
