import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Importing organized faces_dataset
data_dir = 'faces_dataset'
actors = sorted(os.listdir(data_dir))
num_actors = len(actors)

# Load and preprocess data
images = []
labels_detection = []  # Bounding box coordinates
labels_recognition = []  # Actor's name

for actor_idx, actor in enumerate(actors):
    actor_dir = os.path.join(data_dir, actor)
    images_list = os.listdir(actor_dir)
    
    for image_filename in images_list:
        image_path = os.path.join(actor_dir, image_filename)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        
        images.append(image_array)
        labels_detection.append([0.0, 0.0, 1.0, 1.0])  # Assuming a dummy bounding box
        labels_recognition.append(actor_idx)

images = np.array(images)
labels_detection = np.array(labels_detection)
labels_recognition = np.array(labels_recognition)

# Split data into training and testing sets
X_train, X_test, y_detection_train, y_detection_test, y_recognition_train, y_recognition_test = train_test_split(
    images, labels_detection, labels_recognition, test_size=0.3, random_state=42
)

# Create and compile the model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
x = Conv2D(64, (3, 3), activation='relu')(base_model.output)
x = Flatten()(x)
output_detection = Dense(4, activation='linear', name='detection')(x)
output_recognition = Dense(num_actors, activation='softmax', name='recognition')(x)

model = Model(inputs=base_model.input, outputs=[output_detection, output_recognition])

model.compile(
    optimizer=Adam(),
    loss={'detection': 'mean_squared_error', 'recognition': 'sparse_categorical_crossentropy'},
    metrics={'detection': 'mse', 'recognition': 'accuracy'}
)

# Train the model
model.fit(
    X_train, {'detection': y_detection_train, 'recognition': y_recognition_train},
    validation_data=(X_test, {'detection': y_detection_test, 'recognition': y_recognition_test}),
    epochs=5, batch_size=16
)

# Evaluate the model
test_loss, test_detection_loss, test_recognition_loss, test_detection_mse, test_recognition_accuracy = model.evaluate(
    X_test, {'detection': y_detection_test, 'recognition': y_recognition_test}
)

print("Test Detection Loss:", test_detection_loss)
print("Test Recognition Loss:", test_recognition_loss)
print("Test Detection MSE:", test_detection_mse)
print("Test Recognition Accuracy:", test_recognition_accuracy)


# Save the model
model.save('trained_model.h5')



# jwgduygfsqjvdkvwkwdbkwhdkugwkbduwbjhvd
# import random
# import matplotlib.pyplot as plt

# plt.subplot(1, 2, 1)
# plt.imshow(random_image)
# plt.title("Original Image")

# plt.subplot(1, 2, 2)
# plt.imshow(random_mask)
# plt.title("Masked Image")

# plt.show()


# def dice_coefficient(y_true, y_pred):
#     intersection = K.sum(y_true * y_pred)
#     dice = (2. * intersection + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())
#     return dice

# def dice_loss(y_true, y_pred):
#     return 1 - dice_coefficient(y_true, y_pred)