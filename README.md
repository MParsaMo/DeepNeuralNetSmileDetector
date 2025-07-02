# 😊 Emotion Classifier using Neural Networks and Binary Images

This project implements a basic **emotion classification system** using a **fully connected deep neural network (DNN)** with the Keras library. It classifies binary (black & white) face images as either **happy** or **sad** based on their pixel intensities.

---

## 🧠 Project Overview

- 🖼️ Input: Binary face images (1-bit per pixel)
- 🎯 Output: Two emotion classes — **Happy (1,0)** and **Sad (0,1)**
- 🧮 Model: Deep neural network with 3 fully connected layers
- ⚙️ Frameworks: `Keras`, `TensorFlow`, `Pillow`, `NumPy`
- 📈 Output: One-hot encoded prediction (e.g., `[1. 0.]` for happy)


---

## 📦 Requirements

Install dependencies with:

```bash
pip install pillow numpy tensorflow keras

---

🧪 Training Logic
All images in training_set/ are loaded.

Converted to binary using .convert('1').

Flattened into 1D arrays using image.getdata().

Normalized by dividing by 255.

Labeled using file name:

Starts with "happy" → [1, 0]

Starts with "sad" → [0, 1]

---

from PIL import Image
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

directory = 'training_set/'
pixel_intensities = []
labels = []

for filename in os.listdir(directory):
    image = Image.open(directory+filename).convert('1')
    pixel_intensities.append(list(image.getdata()))
    if filename[0:5] == 'happy':
        labels.append([1, 0])
    elif filename[0:3] == 'sad':
        labels.append([0, 1])

pixel_intensities = np.array(pixel_intensities) / 255.0
labels = np.array(labels)


---

🤖 Model Architecture
Input Layer     : 1024 neurons (1 per pixel)
Hidden Layer 1  : 128 neurons, ReLU
Hidden Layer 2  : 64 neurons, ReLU
Output Layer    : 2 neurons, Softmax

---

#Compilation and Training
model = Sequential()
model.add(Dense(128, input_dim=pixel_intensities.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.fit(pixel_intensities, labels, epochs=100, batch_size=32, verbose=2)

---

# Testing the Model
print("Testing the neural network ....")
test_pixel_intensities = []
test_image1 = Image.open('test_set/test_set/happy_test.png').convert('1')
test_pixel_intensities.append(list(test_image1.getdata()))

test_pixel_intensities = np.array(test_pixel_intensities)/255.0
print(model.predict(test_pixel_intensities).round())

---

Example Output
Testing the neural network ....
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 71ms/step
[[1. 0.]]

---

🙋 Contributing
Feel free to:

Improve the model (e.g., add CNN)

Use grayscale/color images

Expand to more emotions (e.g., angry, surprised, etc.)

Add GUI or API support





