#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install keras_preprocessing


# In[3]:


from pymongo import MongoClient
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import to_categorical
from keras_preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image
import io

# MongoDB Connection
client = MongoClient("mongodb+srv://myrnadinorah460:mS37UmTkQunn8Bo2@cluster0.knxjs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["emotion_recognition"]
collection = db["images"]

images = []
labels = []

for document in collection.find():
    image_binary = document["image"]
    label = document["label"]

    # Decode image
    img = Image.open(io.BytesIO(image_binary)).convert("L")  # Grayscale
    img = img.resize((48, 48))
    images.append(img_to_array(img))
    labels.append(label)

# Convert data to numpy arrays
images = np.array(images).reshape(len(images), 48, 48, 1) / 255.0
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded, num_classes=7)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=128)

# Save the model and encoder
model.save("emotion_model.h5")
np.save("label_encoder_classes.npy", le.classes_)
print("Model training complete and saved.")

