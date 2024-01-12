--- ---

# Convolutional Neural Network (CNN) for Fashion MNIST Classification

<div align="center">
  <br>
      <img src="https://github.com/RJohnPaul/Fashion_Mnist_Models/blob/ea69d23ebf38b0608e666d313f7ebf1aee9d751e/Frame%2015.png" alt="Project Banner">
  </br>
</div>

<div align="center">
  <br>
      <img src="https://github.com/RJohnPaul/Fashion_Mnist_Models/blob/265624bd37bdec443e62add66ccdd452761a0d6b/Frame-5(3).png" alt="Project Banner">
  </br>
</div>
</br>


This repo contains TensorFlow code for building and train Convolutional Neural Networks (CNNs) on the Fashion MNIST dataset. The code is provided in two models, one using a simple Dense neural network and the other using a Convolutional Neural Network.

## Table of Contents

- [Dense Neural Network](#dense-neural-network)
- [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
- [Visualization of Convolutional Layers](#visualization-of-convolutional-layers)

## Dense Neural Network

```python
import tensorflow as tf

# Load Fashion MNIST dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

# Build and compile the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=5)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy * 100}%')
```

## Convolutional Neural Network (CNN)

```python
import tensorflow as tf

# Load Fashion MNIST dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images.reshape(60000, 28, 28, 1) / 255.0
test_images = test_images.reshape(10000, 28, 28, 1) / 255.0

# Build and compile the CNN model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the CNN model
model.fit(training_images, training_labels, epochs=5)

# Evaluate the CNN model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy * 100}%')
```

## Visualization of Convolutional Layers

This section of the code visualizes the activation of convolutional layers using matplotlib. It generates a 3x4 grid of images for the first, second, and third images in the test set.

```python
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3, 4)
FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28
CONVOLUTION_NUMBER = 6

# Extract layer outputs for visualization
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

# Visualize convolutional layer activations
for x in range(0, 4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0, x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1, x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2, x].grid(False)
```

<div align="center">
  <br>
      <img src="https://github.com/RJohnPaul/Fashion_Mnist_Models/blob/265624bd37bdec443e62add66ccdd452761a0d6b/Frame-5(3).png" alt="Project Banner">
  </br>
</div>
</br>


---
