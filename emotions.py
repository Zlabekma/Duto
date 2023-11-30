# %% 
import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
# paths 
this_directory = os.path.dirname(os.path.realpath(__file__))
train_directory = os.path.join(this_directory, "train")
test_directory = os.path.join(this_directory, "test")

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Function to load and preprocess data
def load_data(directory):
    data = []
    labels = []
    for label in emotion_labels:
        label_directory = os.path.join(directory, label)
        for image_file in os.listdir(label_directory):
            image = cv2.resize(cv2.imread(os.path.join(label_directory, image_file), cv2.IMREAD_GRAYSCALE), (48, 48))
            data.append(image)
            labels.append(emotion_labels.index(label))
    data, labels = np.array(data), np.array(labels)
    data = data / 255.0
    labels = tf.keras.utils.to_categorical(labels, num_classes=7)
    return data, labels

# Load training data
train_data, train_labels = load_data(train_directory)

# %% 

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

# %%
# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
history = model.fit(datagen.flow(train_data.reshape(-1, 48, 48, 1), train_labels, batch_size=32),
                    steps_per_epoch=len(train_data) / 32,
                    epochs=50)

# %% 
# Load test data
test_data, test_labels = load_data(test_directory)

# Make predictions on test data
predictions = model.predict(test_data.reshape(-1, 48, 48, 1))

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_data[i], cmap='gray')
    predicted_label = np.argmax(predictions[i])
    actual_label = np.argmax(test_labels[i])
    color = 'blue' if predicted_label == actual_label else 'red'
    plt.xlabel(f'Pred: {emotion_labels[predicted_label]} \n Actual: {emotion_labels[actual_label]}', color=color)
plt.show()

evaluation = model.evaluate(test_data.reshape(-1, 48, 48, 1), test_labels)

print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train'], loc='upper left')
plt.show()
