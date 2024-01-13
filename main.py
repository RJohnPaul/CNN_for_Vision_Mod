import tensorflow as tf
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
training_images = training_images / 255.0
test_images = test_images / 255.0

# Define the model for basic image classification
basic_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train the basic model
basic_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
basic_model.fit(training_images, training_labels, epochs=5)

# Evaluate the basic model on the test set
test_loss, test_accuracy = basic_model.evaluate(test_images, test_labels)
print('Basic Model - Test Loss: {}, Test Accuracy: {}'.format(test_loss, test_accuracy * 100))

# Define the model for convolutional neural network (CNN)
cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and summarize the CNN model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()

# Train the CNN model
cnn_model.fit(training_images.reshape(-1, 28, 28, 1), training_labels, epochs=5)

# Evaluate the CNN model on the test set
test_loss, test_accuracy = cnn_model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
print('CNN Model - Test Loss: {}, Test Accuracy: {}'.format(test_loss, test_accuracy * 100))

# Visualize intermediate activations of CNN layers
f, axarr = plt.subplots(3, 4)
FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28
CONVOLUTION_NUMBER = 6

# Create an activation model for visualization
activation_model = tf.keras.models.Model(inputs=cnn_model.input, outputs=[layer.output for layer in cnn_model.layers])

for x in range(0, 4):
    # Predict and visualize activations for the first image
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0, x].grid(False)

    # Predict and visualize activations for the second image
    f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1, x].grid(False)

    # Predict and visualize activations for the third image
    f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2, x].grid(False)

plt.show()

