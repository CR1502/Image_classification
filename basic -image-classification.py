# TensorFlow and tf
from typing import List

import tensorflow as tf

# Import the helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Import the Fashion MNIST dataset (An inbuilt dataset)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_label), (test_images, test_label) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Exploring the data
print(train_images.shape)
print(len(train_label))
print(train_label)
print(test_images.shape)
print(len(test_label))

"""
# Preprocessing the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid = False
plt.show()

This plotted only the first image and the pixel values were in the range of 0 and 255
So we change things up by doing the following
"""

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid = False
    plt.imshow(train_images[i], cmap=plt.cm.binary)  # plt.cm.binary makes the plot b/w
    plt.xlabel(class_names[train_label[i]])
plt.show()

# Now we build the model

""" 
    Model is built by setting up layers.
    Layers extract representations from the data being fed into them
"""
# Setting up the layers

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # We transform the format of the images from a 2D array to a 1D array.
    tf.keras.layers.Dense(128, activation='relu'),  # REtified Linear Unit is an activation function
    # The first layer has 128 nodes or neurons
    tf.keras.layers.Dense(10)  # This layer returns logits array with length of 10
])

# Compiling the model
"""
    Loss function — This measures how accurate the model is during training. You want to minimize this function to 
    "steer" the model in the right direction.
    Optimizer — This is how the model is updated based on the data it sees and its loss function.
    Metrics — Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the 
    images that are correctly classified.

"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training the model

# Feed the model
"""
    Feeding the model with the train_images adn train_labels using model.fit
"""

model.fit(train_images, train_label, epochs=10)

# Evaluate Accuracy

test_loss, test_accuracy = model.evaluate(test_images, test_label, verbose=2)
print("\nTest accuracy: ", test_accuracy)
