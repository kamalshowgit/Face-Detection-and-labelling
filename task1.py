import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load your trained model
model_path = 'trained_model.h5' 
loaded_model = tf.keras.models.load_model(model_path)

# Load and preprocess the testing image
test_image_path = 'training_images/real_00003.jpg'  # Replace with the actual path
test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
test_image_array = tf.keras.preprocessing.image.img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0)

# Use the loaded model for predictions
detection_predictions, recognition_predictions = loaded_model.predict(test_image_array)

# Perform face detection
detected_face_bbox = detection_predictions[0]  # Assuming a single image in batch
detected_face_bbox *= 224  # Rescale the bounding box to match image dimensions

# Get the actor's name
predicted_actor_idx = np.argmax(recognition_predictions[0])
actors = sorted(os.listdir('faces_dataset'))  # Replace with the actual path
predicted_actor_name = actors[predicted_actor_idx]

# Display the results (you might need additional libraries for this)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Display the image
plt.imshow(test_image)
ax = plt.gca()

# Create a Rectangle patch
rect = patches.Rectangle(
    (detected_face_bbox[0], detected_face_bbox[1]),
    detected_face_bbox[2] - detected_face_bbox[0],
    detected_face_bbox[3] - detected_face_bbox[1],
    linewidth=1, edgecolor='r', facecolor='none'
)

# Add the patch to the Axes
ax.add_patch(rect)

# Display the predicted actor's name
plt.text(
    detected_face_bbox[0], detected_face_bbox[1] - 10,
    predicted_actor_name, color='red', backgroundcolor='black'
)

plt.show()
