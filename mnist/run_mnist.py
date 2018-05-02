#!/usr/bin/env python3

# Increasing the robustness of deep neural networks for text classification by examining adversarial examples (code of master thesis)
# Copyright (C) 2017,2018  Marcus Soll
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import keras.utils.np_utils
import keras.datasets
import keras.models
import keras.layers
import keras.losses
import keras.metrics
import keras.optimizers
import keras.backend
import numpy

import tensorflow

import os
import random
import sys

import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.gridspec


def get_gradient(network: keras.models.Sequential, X: numpy.array) -> numpy.array:

    x = tensorflow.placeholder(tensorflow.float32, (None, 28, 28, 1))

    prediction = network(x)

    y_shape = tensorflow.shape(prediction)
    classes = y_shape[1]
    index = tensorflow.argmax(prediction, axis=1)
    target = tensorflow.one_hot(index, classes, on_value=1.0, off_value=0.0)

    logits, = prediction.op.inputs
    loss = tensorflow.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits)
    gradient, = tensorflow.gradients(loss, x)

    return session.run(gradient, feed_dict={x: [X], keras.backend.learning_phase(): 0})


# Set session

session = tensorflow.InteractiveSession()
keras.backend.set_session(session)

# Preprocess dataset

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

y_train = keras.utils.np_utils.to_categorical(y_train, 10)
y_test = keras.utils.np_utils.to_categorical(y_test, 10)

adversarial_test_index = random.randint(0, len(x_train)-1)

# Build or load network

if not os.path.exists('mnist.h5'):
    # Build network
    network = keras.models.Sequential()
    network.add(keras.layers.convolutional.Conv2D(filters=20, kernel_size=(4, 4), activation='relu', input_shape=(28, 28, 1)))
    network.add(keras.layers.convolutional.Conv2D(filters=40, kernel_size=(4, 4), activation='relu'))
    network.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    network.add(keras.layers.Dropout(0.1))
    network.add(keras.layers.convolutional.Conv2D(filters=60, kernel_size=(3, 3), activation='relu'))
    network.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    network.add(keras.layers.Flatten())
    network.add(keras.layers.Dropout(0.1))
    network.add(keras.layers.Dense(300, activation='relu'))
    network.add(keras.layers.Dense(150, activation='relu'))
    network.add(keras.layers.Dropout(0.1))
    network.add(keras.layers.Dense(10, activation='relu'))
    network.add(keras.layers.Activation('softmax'))

    network.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=[keras.metrics.categorical_accuracy])

    network.fit(x_train, y_train, epochs=5, verbose=1, batch_size=128)
    network.save('mnist.h5')
else:
    network = keras.models.load_model('mnist.h5')

# Evaluate network

# result = network.evaluate(x_test, y_test, batch_size=256, verbose=0)

# print()
# print('Loss: {}'.format(result[0]))
# print('Accuracy: {}'.format(result[1]))

# Get gradient

print('Searching for adversarial', end='')

gradient = get_gradient(network, x_train[adversarial_test_index])

# Calculate adversarial

gradient_sign = numpy.sign(gradient)

found = False
original_image = x_train[adversarial_test_index]
adversarial = x_train[adversarial_test_index]
source_index = numpy.argmax(y_train[adversarial_test_index])
target_index = None

while not found:
    print('.', end='')
    sys.stdout.flush()
    adversarial = adversarial + 0.01 * gradient_sign
    adversarial = numpy.clip(adversarial, 0.0, 1.0)
    prediction = network.predict([adversarial])
    target_index = numpy.argmax(prediction[0])
    found = (target_index != source_index) and (prediction[0][target_index] > 0.1)

print()
print('Found adversarial: {} -> {}'.format(source_index, target_index))

# Save result

confidence_target = network.predict([adversarial])
confidence_target = confidence_target[0][target_index]
confidence_source = network.predict(numpy.array([x_train[adversarial_test_index]]))
confidence_source = confidence_source[0][source_index]

figure = matplotlib.pyplot.figure(figsize=(10, 2))
grid = matplotlib.gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

subfigure = figure.add_subplot(grid[0, 0])
subfigure.imshow(numpy.squeeze(x_train[adversarial_test_index]), cmap='gray', interpolation='none')
subfigure.set_xticks([])
subfigure.set_yticks([])
subfigure.set_xlabel('Original image: {} ({})'.format(source_index, confidence_source))

subfigure = figure.add_subplot(grid[0, 1])
subfigure.imshow(numpy.squeeze(adversarial), cmap='gray', interpolation='none')
subfigure.set_xticks([])
subfigure.set_yticks([])
subfigure.set_xlabel('Adversarial example: {} ({})'.format(target_index, confidence_target))

grid.tight_layout(figure)
matplotlib.pyplot.savefig("adversarial.png")

# kate: replace-tabs true; indent-width 4; indent-mode python;
