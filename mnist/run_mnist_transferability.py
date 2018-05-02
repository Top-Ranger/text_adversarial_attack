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

import datetime
import json
import os
import math
import random

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.gridspec

NUMBER_TESTED = 500
PRINT_GENERATED_IMAGES = True

session = tensorflow.InteractiveSession()
keras.backend.set_session(session)

def get_single_layer_network(kernels=[3, 4, 5]):
    input_tensor = keras.layers.Input(shape=(28, 28, 1))
    kernel_list = []
    for kernel in kernels:
        single_kernel = keras.models.Sequential()
        single_kernel = keras.layers.Convolution2D(filters=25, kernel_size=kernel, activation='relu')(input_tensor)
        single_kernel = keras.layers.MaxPool2D(pool_size=(kernel, kernel))(single_kernel)#
        single_kernel = keras.layers.Flatten()(single_kernel)
        kernel_list.append(single_kernel)

    output_tensor = keras.layers.Concatenate()(kernel_list)
    convolutional_filters = keras.models.Model(outputs=output_tensor, inputs=input_tensor)

    network = keras.models.Sequential()
    network.add(convolutional_filters)
    network.add(keras.layers.Dropout(0.5))
    network.add(keras.layers.Dense(10))
    network.add(keras.layers.Activation('softmax'))

    return network


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


def get_adversarial_example(input_image: numpy.array, network: keras.models.Sequential) -> numpy.array:
    i = 0
    found = False
    source_index = numpy.argmax(network.predict(numpy.array([input_image])))
    gradient = get_gradient(network, input_image)
    gradient_sign = numpy.sign(gradient)
    adversarial = input_image
    while not found:
        i += 1
        # Abort if no adversarial can be found
        if i > 100:
            return []
        #print('Searching for adversarial example: {0:3d}'.format(i), end='\r')
        adversarial = adversarial + 0.01 * gradient_sign
        adversarial = numpy.clip(adversarial, 0.0, 1.0)
        prediction = network.predict([adversarial])
        target_index = numpy.argmax(prediction[0])
        found = (target_index != source_index) and (prediction[0][target_index] > 0.1)
    #print()
    return adversarial


def get_splitted_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    y_train = keras.utils.np_utils.to_categorical(y_train, 10)
    y_test = keras.utils.np_utils.to_categorical(y_test, 10)

    split = len(x_train)//2
    x_train_1 = x_train[:split]
    x_train_2 = x_train[split:]
    y_train_1 = y_train[:split]
    y_train_2 = y_train[split:]

    return (x_train, y_train), (x_train_1, y_train_1), (x_train_2, y_train_2), (x_test, y_test)


(x_train_complete, y_train_complete), (x_train_1, y_train_1), (x_train_2, y_train_2), (x_test, y_test) = get_splitted_dataset()

# Target network
if not os.path.exists('mnist-target.h5'):
    network_target = get_single_layer_network()
    network_target.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=[keras.metrics.categorical_accuracy])
    network_target.fit(x_train_1, y_train_1, epochs=10, verbose=1, batch_size=128)

    print('Evaluating model')
    result = network_target.evaluate(x_test, y_test, batch_size=128, verbose=0)
    print()
    print('Loss: {}'.format(result[0]))
    print('Accuracy: {}'.format(result[1]))

    metadata = dict()
    metadata['loss'] = result[0]
    metadata['accuracy'] = result[1]
    metadata['dates'] = str(datetime.datetime.today())
    with open('./mnist-target.json', 'w') as file:
        json.dump(metadata, file, indent=4,)

    network_target.save('./mnist-target.h5')
else:
    network_target = keras.models.load_model('mnist-target.h5')

# Retrained network
if not os.path.exists('mnist-retrained.h5'):
    network_retrained = get_single_layer_network()
    network_retrained.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=[keras.metrics.categorical_accuracy])
    network_retrained.fit(x_train_1, y_train_1, epochs=10, verbose=1, batch_size=128)

    print('Evaluating model')
    result = network_retrained.evaluate(x_test, y_test, batch_size=128, verbose=0)
    print()
    print('Loss: {}'.format(result[0]))
    print('Accuracy: {}'.format(result[1]))

    metadata = dict()
    metadata['loss'] = result[0]
    metadata['accuracy'] = result[1]
    metadata['dates'] = str(datetime.datetime.today())
    with open('./mnist-retrained.json', 'w') as file:
        json.dump(metadata, file, indent=4,)

    network_retrained.save('./mnist-retrained.h5')
else:
    network_retrained = keras.models.load_model('mnist-retrained.h5')

# Different dataset network
if not os.path.exists('mnist-dataset.h5'):
    network_dataset = get_single_layer_network()
    network_dataset.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=[keras.metrics.categorical_accuracy])
    network_dataset.fit(x_train_1, y_train_1, epochs=10, verbose=1, batch_size=128)

    print('Evaluating model')
    result = network_dataset.evaluate(x_test, y_test, batch_size=128, verbose=0)
    print()
    print('Loss: {}'.format(result[0]))
    print('Accuracy: {}'.format(result[1]))

    metadata = dict()
    metadata['loss'] = result[0]
    metadata['accuracy'] = result[1]
    metadata['dates'] = str(datetime.datetime.today())
    with open('./mnist-dataset.json', 'w') as file:
        json.dump(metadata, file, indent=4,)

    network_dataset.save('./mnist-dataset.h5')
else:
    network_dataset = keras.models.load_model('mnist-dataset.h5')

# Different kernel
if not os.path.exists('mnist-kernel.h5'):
    network_kernel = get_single_layer_network([3, 3, 5, 5])
    network_kernel.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=[keras.metrics.categorical_accuracy])
    network_kernel.fit(x_train_1, y_train_1, epochs=10, verbose=1, batch_size=128)

    print('Evaluating model')
    result = network_kernel.evaluate(x_test, y_test, batch_size=128, verbose=0)
    print()
    print('Loss: {}'.format(result[0]))
    print('Accuracy: {}'.format(result[1]))

    metadata = dict()
    metadata['loss'] = result[0]
    metadata['accuracy'] = result[1]
    metadata['dates'] = str(datetime.datetime.today())
    with open('./mnist-kernel.json', 'w') as file:
        json.dump(metadata, file, indent=4,)

    network_kernel.save('./mnist-kernel.h5')
else:
    network_kernel = keras.models.load_model('mnist-kernel.h5')

successful_retrained = 0
successful_dataset = 0
successful_kernel = 0

tested = set()
used_index = []

if PRINT_GENERATED_IMAGES:
    if not os.path.exists('./generated_adversarials_images/'):
        os.makedirs('./generated_adversarials_images/')

for number in range(NUMBER_TESTED):
    print("Testing adversarial number {0:4d} of {1:4d}".format(number+1, NUMBER_TESTED), end='\r')
    found = False

    while not found:
        index = random.randint(0, len(x_train_complete))
        if index in tested:
            continue
        tested.add(index)
        target_class = numpy.argmax(y_train_complete[index])
        found = target_class == numpy.argmax(network_target.predict(numpy.asarray([x_train_complete[index]]))) and \
                target_class == numpy.argmax(network_retrained.predict(numpy.asarray([x_train_complete[index]]))) and \
                target_class == numpy.argmax(network_dataset.predict(numpy.asarray([x_train_complete[index]]))) and \
                target_class == numpy.argmax(network_kernel.predict(numpy.asarray([x_train_complete[index]])))

    adversarial = get_adversarial_example(x_train_complete[index], network_target)
    if len(adversarial) == 0:
        print('Warning: No adversarial found for index {}'.format(index))
        continue
    if target_class != numpy.argmax(network_retrained.predict(numpy.asarray([x_train_complete[index]]))):
        successful_retrained = successful_retrained + 1
    if target_class != numpy.argmax(network_dataset.predict(numpy.asarray([x_train_complete[index]]))):
        successful_dataset = successful_dataset + 1
    if target_class != numpy.argmax(network_kernel.predict(numpy.asarray([x_train_complete[index]]))):
        successful_kernel = successful_kernel + 1
    used_index.append(index)

    if PRINT_GENERATED_IMAGES:
        figure = matplotlib.pyplot.figure(figsize=(10, 2))
        grid = matplotlib.gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

        subfigure = figure.add_subplot(grid[0, 0])
        subfigure.imshow(numpy.squeeze(x_train_complete[index]), cmap='gray', interpolation='none')
        subfigure.set_xticks([])
        subfigure.set_yticks([])
        subfigure.set_xlabel('Original image: {}'.format(numpy.argmax(y_train_complete[index])))

        subfigure = figure.add_subplot(grid[0, 1])
        subfigure.imshow(numpy.squeeze(adversarial), cmap='gray', interpolation='none')
        subfigure.set_xticks([])
        subfigure.set_yticks([])
        subfigure.set_xlabel('Adversarial example: {}'.format(numpy.argmax(network_target.predict([adversarial]))))

        grid.tight_layout(figure)
        matplotlib.pyplot.savefig("./generated_adversarials_images/{}.png".format(number+1))
        matplotlib.pyplot.close('all')

print()

print('--- Results ---')
print('Transferability:')
print('      Retrained first half: {}'.format(successful_retrained / NUMBER_TESTED))
print('      Second half: {}'.format(successful_dataset / NUMBER_TESTED))
print('      Alternative kernels first half: {}'.format(successful_kernel / NUMBER_TESTED))

result = dict()
result['retrained_rate'] = successful_retrained / NUMBER_TESTED
result['dataset_rate'] = successful_dataset / NUMBER_TESTED
result['kernel_rate'] = successful_kernel / NUMBER_TESTED
result['successful_retrained'] = successful_retrained
result['successful_dataset'] = successful_dataset
result['successful_kernel'] = successful_kernel
result['number_tested'] = NUMBER_TESTED

with open('./mnist_transferability.json', 'w') as file:
    json.dump(result, file, indent=4)

# kate: replace-tabs true; indent-width 4; indent-mode python;
