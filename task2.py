import os
import cv2
import pandas as pd

# Task 1: Read/import images from folder 'training_images'
image_folder = 'training_images'
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Initialize a list to store face metadata
face_data = []

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Task 2: Loop through images and detect faces
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    
    # Task 3: Extract metadata and store in the face_data list
    for (x, y, w, h) in faces:
        face_metadata = {
            'ImageFile': image_file,
            'FaceX': x,
            'FaceY': y,
            'FaceWidth': w,
            'FaceHeight': h
        }
        face_data.append(face_metadata)

# Task 4: Create a DataFrame from the face data and save as CSV
face_df = pd.DataFrame(face_data)
output_csv = 'face_metadata.csv'
face_df.to_csv(output_csv, index=False)

print("Face detection and metadata extraction completed.")
