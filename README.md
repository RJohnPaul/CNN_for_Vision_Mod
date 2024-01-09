# TensorFlow Fashion MNIST Models

This repository contains a TensorFlow script (`fashion_mnist_models.py`) for building and training two models on the Fashion MNIST dataset. The script first creates a basic neural network and then extends it to a Convolutional Neural Network (CNN). 

The script loads the Fashion MNIST dataset, normalizes pixel values, and constructs a sequential neural network model with hidden and output layers. The model is compiled using the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric. After training for 5 epochs, the script evaluates the model on the test data, printing the test loss and accuracy.

Following this, the script reloads the dataset, reshapes and normalizes pixel values for the CNN model. The CNN model is then built with Conv2D, MaxPooling2D, Flatten, and Dense layers. The CNN model is compiled, its summary is displayed, and it is trained for 5 epochs. The script evaluates the CNN model on the test data, printing the results.

For visual inspection, the script utilizes Matplotlib to create subplots for visualizing convolutional activations. It creates an activation model to get intermediate layer outputs and visualizes convolutional activations for three different images.

## Instructions:

1. **Clone the repository:** `git clone https://github.com/RJohnPaul/CNN_for_Vision_Mod.git`

## Requirements:

- [TensorFlow](https://www.tensorflow.org/install): `pip install tensorflow`
- [Matplotlib](https://matplotlib.org/stable/users/installing.html): `pip install matplotlib`

## Run the Code:

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Run the script:** `python fashion_mnist_models.py`

## Features:

- **Modular Code:** Easily understand and modify the script for your own projects.
- **Versatile Models:** Neural network and Convolutional Neural Network models for image classification.
- **Visualization:** Convolutional activation visualizations for enhanced understanding.
- **Educational:** Suitable for learning TensorFlow basics and CNN concepts.

## Results:

After running the script, observe the test loss and accuracy for both the neural network and CNN models. Additionally, visualizations of CNN activations for three different images are provided.

## Modules Required:

- **TensorFlow:** The open-source machine learning library. [Installation](https://www.tensorflow.org/install)

- **Matplotlib:** A comprehensive library for creating static, animated, and interactive visualizations. [Installation](https://matplotlib.org/stable/users/installing.html)

## License:

This project is licensed under the [MIT License](LICENSE).

---
