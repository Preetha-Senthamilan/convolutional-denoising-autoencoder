# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
An autoencoder is an unsupervised neural network that encodes input images into lower-dimensional representations and decodes them back, aiming for identical outputs. We use the MNIST dataset, consisting of 60,000 handwritten digits (28x28 pixels), to train a convolutional neural network for digit classification. The goal is to accurately classify each digit into one of 10 classes, from 0 to 9.

![image](https://github.com/user-attachments/assets/6fbdbd3a-8874-4425-8e18-04cc41beb9ce)

## Convolution Autoencoder Network Model

![image](https://github.com/user-attachments/assets/05ded2c6-ea8e-47e8-a5e7-bcc445b5c72f)


## DESIGN STEPS

### STEP 1:
Import the necessary libraries and dataset.

### STEP 2:
Load the dataset and scale the values for easier computation.

### STEP 3:
Add noise to the images randomly for both the train and test sets.

### Step 4:
Build the Neural Model using,Convolutional,Pooling and Upsampling layers.

### Step 5:
Pass test data for validating manually.

### Step 6:
Plot the predictions for visualization.


## PROGRAM
### Name:PREETHA S
### Register Number:212222230110
```
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))
noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape)
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
print('Name: SWETHA S           Register Number: 212222230155       ')
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
decoded_imgs = autoencoder.predict(x_test_noisy)
n = 10
print('Name: PREETHA S          Register Number: 212222230110')

plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original images
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy images
    ax = plt.subplot(3, n, i + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstructed images
    ax = plt.subplot(3, n, i + 2 * n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
history = autoencoder.fit(x_train_noisy, x_train_scaled,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test_noisy, x_test_scaled))
import pandas as pd
metrics=pd.DataFrame(autoencoder.history.history)
plt.figure(figsize=(7,2.5))
plt.plot(metrics['loss'], label='Training Loss')
plt.plot(metrics['val_loss'], label='Validation Loss')
plt.title('Training Loss vs. Validation Loss\nPREETHA S - 212222230110')
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot


![Screenshot 2024-11-17 112742](https://github.com/user-attachments/assets/b0abb924-bfb3-461b-80ea-e3a13c2ed24d)



### Original vs Noisy Vs Reconstructed Image


![Screenshot 2024-11-17 112427](https://github.com/user-attachments/assets/8e0fb189-c040-4da6-b130-d350f4d57ce8)



## RESULT

Thus we have successfully developed a convolutional autoencoder for image denoising application.
