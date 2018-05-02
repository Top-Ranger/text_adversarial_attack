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

import os
import json
import datetime
import enum
import tempfile

import network
import dataset
import helpers

import numpy
import keras.utils.np_utils
import keras.backend
import keras.layers
import keras.models
import tensorflow


"""
This variable holds the percentage of the contribution of the hard label loss to the overall loss for the defensive distillation training.
It should be relatively small

See: Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. CoRR, abs/1503.02531, March 2015. NIPS 2014 Deep Learning Workshop.
"""
DISTILLATION_LOSS_HARD_LABEL_PERCENTAGE = 0.1

def get_network(dataset_type: dataset.DatasetType, encoding_type: dataset.Encoding, kernel_variation=[3, 4, 5], every_xth_trainings_data=1, skip_trainings_data=0, ignore_cached=False, cache_prefix=''):
    """
    Returns a trained network.

    The network is cached to disk to avoid retraining. If a cached network is found, it will be used. Statistics about the network can be found in the metadata file besides the cached network (in ./models)
    :param dataset_type: The dataset which should be used.
    :type dataset_type: DatasetType
    :param encoding_type: The encoding for the dataset.
    :type encoding_type: dataset.Encoding
    :param kernel_variation: Kernel variation used to create the network. Each entry equals to 25 kernels.
    :type kernel_variation: list of integers
    :param: every_xth_trainings_data: Defines which part of the data set is used, e.g. setting it to 1 will keep each data point, setting it to 2 will keep each second data point.
    :type: every_xth_trainings_data: int
    :param skip_trainings_data: Sets how many data points are skipped at the beginning. Setting it to 0 includes the whole data set.
    :type skip_trainings_data: int
    :param ignore_cached: If this is set to True, the network will always be trained ignoring cached networks if existing
    :type ignore_cached: bool
    :param cache_prefix: Prefix for disk cache. Allows to use different networks with the same parameter
    :type cache_prefix: str
    :returns: keras.Sequential Tained network
    """
    # Try loading cached
    if not ignore_cached and os.path.exists('./models/{}-{}-{}trained-{}-{}-{}.h5'.format(dataset_type.value, encoding_type.value, cache_prefix, str(kernel_variation), every_xth_trainings_data, skip_trainings_data)):
        return keras.models.load_model('./models/{}-{}-{}trained-{}-{}-{}.h5'.format(dataset_type.value, encoding_type.value, cache_prefix, str(kernel_variation), every_xth_trainings_data, skip_trainings_data))

    # Create dataset
    if dataset.is_cached('{}-{}'.format(dataset_type.value, encoding_type.value)):
        print('Using cached dataset')
        (x_train, y_train), (x_test, y_test), class_labels = dataset.get_from_cache('{}-{}'.format(dataset_type.value, encoding_type.value), memmap='r')
    else:
        print('Creating dataset')
        (x_train, y_train), (x_test, y_test), class_labels = dataset.get_standard_dataset(dataset_type, encoding_type)
        dataset.cache_dataset(x_train, y_train, x_test, y_test, class_labels, '{}-{}'.format(dataset_type.value, encoding_type.value))

    # Filter datasets
    x_train_used = list()
    y_train_used = list()

    for i in range(skip_trainings_data, len(x_train)):
        if (i-skip_trainings_data)%every_xth_trainings_data == 0:
            x_train_used.append(x_train[i])
            y_train_used.append(y_train[i])

    x_train_used = numpy.asarray(x_train_used)
    y_train_used = numpy.asarray(y_train_used)

    # Create and train model
    print('Getting model')
    representation_size = {
        dataset.Encoding.WORD2VEC.value: 300,
        dataset.Encoding.CHARACTER.value: helpers.number_character_index_dict(),
    }[encoding_type.value]
    model = network.get_single_layer_network(classes=len(class_labels), representation_size=representation_size, kernels=kernel_variation)

    y_train_used = keras.utils.np_utils.to_categorical(y_train_used, len(class_labels))
    y_test = keras.utils.np_utils.to_categorical(y_test, len(class_labels))

    print('Training model')
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=[keras.metrics.categorical_accuracy])
    model.fit(x_train_used, y_train_used, epochs=10, verbose=1, batch_size=128)

    print('Evaluating model')
    result = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    print()
    print('Loss: {}'.format(result[0]))
    print('Accuracy: {}'.format(result[1]))

    if not os.path.exists('./models/'):
        os.makedirs('./models/')

    # Save model and metadata
    metadata = dict()
    metadata['trainings_dataset'] = {'dataset': dataset_type.value, 'text encoding': encoding_type.value, 'every_xth_trainings_data': every_xth_trainings_data, 'skip_trainings_data': skip_trainings_data, 'size': len(x_train_used)}
    metadata['loss'] = result[0]
    metadata['accuracy'] = result[1]
    metadata['class_labels'] = class_labels
    metadata['date'] = str(datetime.datetime.today())
    metadata['kernels'] = str(kernel_variation)
    with open('./models/{}-{}-{}trained-{}-{}-{}.metadata'.format(dataset_type.value, encoding_type.value, cache_prefix, str(kernel_variation), every_xth_trainings_data, skip_trainings_data), 'w') as file:
        json.dump(metadata, file, indent=4,)

    model.save('./models/{}-{}-{}trained-{}-{}-{}.h5'.format(dataset_type.value, encoding_type.value, cache_prefix, str(kernel_variation), every_xth_trainings_data, skip_trainings_data))

    # Return model
    return model


def get_defensively_distilled_network(dataset_type: dataset.DatasetType, encoding_type: dataset.Encoding, kernel_variation=[3, 4, 5], every_xth_trainings_data=1, skip_trainings_data=0, temperature=20, ignore_cached=False, cache_prefix=''):
    """
    Returns a trained network which was trained with 'defensive distillation'.

    The network is cached to disk to avoid retraining. If a cached network is found, it will be used. Statistics about the network can be found in the metadata file besides the cached network (in ./models)
    :param dataset_type: The dataset which should be used.
    :type dataset_type: DatasetType
    :param encoding_type: The encoding for the dataset.
    :type encoding_type: dataset.Encoding
    :param kernel_variation: Kernel variation used to create the network. Each entry equals to 25 kernels.
    :type kernel_variation: list of integers
    :param: every_xth_trainings_data: Defines which part of the data set is used, e.g. setting it to 1 will keep each data point, setting it to 2 will keep each second data point.
    :type: every_xth_trainings_data: int
    :param skip_trainings_data: Sets how many data points are skipped at the beginning. Setting it to 0 includes the whole data set.
    :type skip_trainings_data: int
    :param temperature: Trainings temperature
    :type temperature: int or float
    :param ignore_cached: If this is set to True, the network will always be trained ignoring cached networks if existing
    :type ignore_cached: bool
    :param cache_prefix: Prefix for disk cache. Allows to use different networks with the same parameter
    :type cache_prefix: str
    :returns: keras.Sequential Tained network
    """
    # Custom loss function implementing weighted average over hard and soft labels
    def loss_function_distillation(truth, prediction):
        """
        Custom loss function following advice from Hinton et al. (2015)

        Hyperparameters: percentage of hard label loss: DISTILLATION_LOSS_HARD_LABEL_PERCENTAGE
        The soft labels are multiplied by temperature^2 as described in Hinton et al. (2015)

        Source: Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. CoRR, abs/1503.02531, March 2015. NIPS 2014 Deep Learning Workshop.

        :param truth: Truth tensor (combined hard and soft labels in that order)
        :type truth: tensor
        :param prediction: Tensor with predictted classes (combined hard and soft labels in that order)
        :type prediction: tensor
        :return: loss
        """
        number_classes = prediction.shape[1] // 2  # The prediction shape is partly known, the truth shape isn't
        truth_hard, truth_soft = truth[:, :number_classes], truth[:, number_classes:]
        prediction_hard, prediction_soft = prediction[:, :number_classes], prediction[:, number_classes:]
        return DISTILLATION_LOSS_HARD_LABEL_PERCENTAGE * keras.losses.categorical_crossentropy(truth_hard, prediction_hard) + (1.0 - DISTILLATION_LOSS_HARD_LABEL_PERCENTAGE) * keras.losses.categorical_crossentropy(truth_soft, prediction_soft) * temperature * temperature

    # Try loading cached
    if not ignore_cached and os.path.exists('./models/defensive_distillation({})-{}-{}-{}trained-{}-{}-{}.h5'.format(temperature, dataset_type.value, encoding_type.value, cache_prefix, str(kernel_variation), every_xth_trainings_data, skip_trainings_data)):
        return keras.models.load_model('./models/defensive_distillation({})-{}-{}-{}trained-{}-{}-{}.h5'.format(temperature, dataset_type.value, encoding_type.value, cache_prefix, str(kernel_variation), every_xth_trainings_data, skip_trainings_data), custom_objects={'loss_function_distillation': loss_function_distillation})

    # Create dataset
    if dataset.is_cached('{}-{}'.format(dataset_type.value, encoding_type.value)):
        print('Using cached dataset')
        (x_train, y_train), (x_test, y_test), class_labels = dataset.get_from_cache('{}-{}'.format(dataset_type.value, encoding_type.value), memmap='r')
    else:
        print('Creating dataset')
        (x_train, y_train), (x_test, y_test), class_labels = dataset.get_standard_dataset(dataset_type, encoding_type)
        dataset.cache_dataset(x_train, y_train, x_test, y_test, class_labels, '{}-{}'.format(dataset_type.value, encoding_type.value))

    # Get teacher
    teacher_model = get_network(dataset_type, encoding_type, kernel_variation=kernel_variation, every_xth_trainings_data=every_xth_trainings_data, skip_trainings_data=skip_trainings_data, ignore_cached=ignore_cached, cache_prefix=cache_prefix)

    # Filter datasets
    x_train_used = list()
    y_train_used = list()

    for i in range(skip_trainings_data, len(x_train)):
        if (i-skip_trainings_data)%every_xth_trainings_data == 0:
            x_train_used.append(x_train[i])
            y_train_used.append(y_train[i])

    x_train_used = numpy.asarray(x_train_used)
    y_train_used = numpy.asarray(y_train_used)

    # Create and train model
    print('Getting model')
    representation_size = {
        dataset.Encoding.WORD2VEC.value: 300,
        dataset.Encoding.CHARACTER.value: helpers.number_character_index_dict(),
    }[encoding_type.value]
    model = network.get_single_layer_network(classes=len(class_labels), representation_size=representation_size, kernels=kernel_variation)

    # Prepare model for training
    logits_new_model = model.layers[-1].output

    soft_targets = keras.layers.Lambda(lambda x: keras.backend.softmax(x / temperature))(logits_new_model)
    hard_targets = keras.layers.Activation('softmax', name='jjj')(logits_new_model)

    new_output = keras.layers.Concatenate()([hard_targets, soft_targets])

    new_model = keras.models.Model(model.input, new_output)

    y_train_used = keras.utils.np_utils.to_categorical(y_train_used, len(class_labels))
    y_test = keras.utils.np_utils.to_categorical(y_test, len(class_labels))

    # Get soft labels
    print('Getting soft labels')
    y_train_used_altered = _get_soft_labels(x_train_used, y_train_used, teacher_model, temperature=temperature)
    y_train_used_altered = numpy.concatenate((y_train_used, y_train_used_altered), axis=1)

    print('Training model')
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=[keras.metrics.categorical_accuracy])
    new_model.compile(loss=loss_function_distillation, optimizer=keras.optimizers.Adadelta(), metrics=[keras.metrics.categorical_accuracy])
    new_model.fit(x_train_used, y_train_used_altered, epochs=10, verbose=1, batch_size=128)

    print('Evaluating model')
    result = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    print()
    print('Loss: {}'.format(result[0]))
    print('Accuracy: {}'.format(result[1]))

    if not os.path.exists('./models/'):
        os.makedirs('./models/')

    # Save model and metadata
    metadata = dict()
    metadata['trainings_dataset'] = {'dataset': dataset_type.value, 'text encoding': encoding_type.value, 'every_xth_trainings_data': every_xth_trainings_data, 'skip_trainings_data': skip_trainings_data, 'size': len(x_train_used)}
    metadata['loss'] = result[0]
    metadata['accuracy'] = result[1]
    metadata['class_labels'] = class_labels
    metadata['date'] = str(datetime.datetime.today())
    metadata['kernels'] = str(kernel_variation)
    metadata['defense'] = 'defensive distillation'
    metadata['temperature'] = temperature
    metadata['hard _label_loss_function_percentage'] = DISTILLATION_LOSS_HARD_LABEL_PERCENTAGE
    with open('./models/defensive_distillation({})-{}-{}-{}trained-{}-{}-{}.metadata'.format(temperature, dataset_type.value, encoding_type.value, cache_prefix, str(kernel_variation), every_xth_trainings_data, skip_trainings_data), 'w') as file:
        json.dump(metadata, file, indent=4,)

    model.save('./models/defensive_distillation({})-{}-{}-{}trained-{}-{}-{}.h5'.format(temperature, dataset_type.value, encoding_type.value, cache_prefix, str(kernel_variation), every_xth_trainings_data, skip_trainings_data))

    # Return model
    return model


def _get_soft_labels(X: numpy.array, Y: numpy.array, model: keras.Model, temperature=20, batch_size=256):
    """
    Calculates the soft labels for a given dataset. The input y values have to be one-hot encoded.

    :param X: X values
    :type X: numpy.array
    :param Y: Class labels, encoded as one-hot.
    :type Y: numby.array
    :param model: The teacher model
    :type model: keras.Model
    :param temperature: The temperature for which the labels should be calculated. Must match the student networks temperature.
    :type temperature: int or float
    :param batch_size: Size of batch for calculation. The larger, the faster but also the more memory is needed.
    :type batch_size: int
    :returns: numpy.array containing soft labels
    """
    # Sanity check
    if int(Y.shape[0]) != int(X.shape[0]):
        raise ValueError('X length is not equal to Y length')

    soft_labels = numpy.memmap(tempfile.TemporaryFile(), dtype='float', mode='w+', shape=Y.shape)

    # Create tensor
    input_shape = list(X.shape)
    input_shape[0] = batch_size  # batch size has to be considered for input size
    input_tensor = tensorflow.placeholder(tensorflow.float32, input_shape)

    prediction = model(input_tensor)
    logits, = prediction.op.inputs
    soft_label_tensor = keras.backend.softmax(logits / temperature)

    session = keras.backend.get_session()

    # Prepare computation
    number_batches = (X.shape[0] // batch_size) + 1

    # Run computation is batches so not everything has to be stored in memory
    for current_batch in range(number_batches):
        print('Calculating soft labels - batch {0:3d} of {1:3d}'.format(current_batch+1, number_batches), end='\r' if current_batch != number_batches-1 else '\n')
        input_data = list()
        for i in range(batch_size*current_batch, batch_size*(current_batch+1)):
            if i < X.shape[0]:
                input_data.append(X[i])
            else:
                input_data.append(numpy.zeros(input_shape[1:]))
        if len(input_data) == 0:
            break
        input_data = numpy.asarray(input_data)

        # Computation
        soft_label_output = session.run(soft_label_tensor, feed_dict={input_tensor: input_data, keras.backend.learning_phase(): 0})

        # Add to result
        for i in range(soft_label_output.shape[0]):
            if i + batch_size*current_batch >= Y.shape[1]:
                # We have all data added - jump out
                break
            for j in range(soft_label_output.shape[1]):
                soft_labels[i + batch_size*current_batch][j] = soft_label_output[i][j]

    # Finally return the soft labels
    return soft_labels


if __name__ == '__main__':
    for dataset_type in dataset.DatasetType:
        model = get_defensively_distilled_network(dataset_type, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0, temperature=20)

# kate: replace-tabs true; indent-width 4; indent-mode python;
