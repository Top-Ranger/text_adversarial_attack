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
import adversarial_dataset
import helpers

import numpy
import keras.utils.np_utils
import keras.backend
import keras.layers
import keras.models
import tensorflow


def get_network(dataset_type: dataset.DatasetType, encoding_type: dataset.Encoding, adversarial_number=-1, kernel_variation=[3, 4, 5], ignore_cached=False, cache_prefix=''):
    """
    Returns a trained network.

    The network is cached to disk to avoid retraining. If a cached network is found, it will be used. Statistics about the network can be found in the metadata file besides the cached network (in ./models)

    :raises: NoAdversarialData

    :param dataset_type: The dataset which should be used.
    :type dataset_type: DatasetType
    :param encoding_type: The encoding for the dataset.
    :type encoding_type: dataset.Encoding
    :param adversarial_number: Number of adversarial data added. Set -1 to add all adversarial examples
    :type adversarial_number: int
    :param kernel_variation: Kernel variation used to create the network. Each entry equals to 25 kernels.
    :type kernel_variation: list of integers
    :param ignore_cached: If this is set to True, the network will always be trained ignoring cached networks if existing
    :type ignore_cached: bool
    :param cache_prefix: Prefix for disk cache. Allows to use different networks with the same parameter
    :type cache_prefix: str
    :returns: keras.Sequential Tained network
    """
    # Try loading cached
    if not ignore_cached and os.path.exists('./models/{}-{}-{}trained-with-adversarial-examples-{}-{}.h5'.format(dataset_type.value, encoding_type.value, cache_prefix, str(kernel_variation), adversarial_number)):
        return keras.models.load_model('./models/{}-{}-{}trained-with-adversarial-examples-{}-{}.h5'.format(dataset_type.value, encoding_type.value, cache_prefix, str(kernel_variation), adversarial_number))

    # Create dataset - always create a new dataset to reduce the effect of randomness
    (x_train, y_train), (x_test, y_test), class_labels = adversarial_dataset.get_standard_dataset(dataset_type, encoding_type, adversarial_number=adversarial_number)

    # Create and train model
    print('Getting model')
    representation_size = {
        dataset.Encoding.WORD2VEC.value: 300,
        dataset.Encoding.CHARACTER.value: helpers.number_character_index_dict(),
    }[encoding_type.value]
    model = network.get_single_layer_network(classes=len(class_labels), representation_size=representation_size, kernels=kernel_variation)

    y_train = keras.utils.np_utils.to_categorical(y_train, len(class_labels))
    y_test = keras.utils.np_utils.to_categorical(y_test, len(class_labels))

    print('Training model')
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=[keras.metrics.categorical_accuracy])
    model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=128)

    print('Evaluating model')
    result = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    print()
    print('Loss: {}'.format(result[0]))
    print('Accuracy: {}'.format(result[1]))

    if not os.path.exists('./models/'):
        os.makedirs('./models/')

    # Save model and metadata
    metadata = dict()
    metadata['trainings_dataset'] = {'dataset': dataset_type.value, 'text encoding': encoding_type.value, 'size': len(x_train), 'normal_number': len(x_train)-adversarial_number, 'adversarial_number': adversarial_number, 'adversarial_percentage': adversarial_number/len(x_train)}
    metadata['loss'] = result[0]
    metadata['accuracy'] = result[1]
    metadata['class_labels'] = class_labels
    metadata['date'] = str(datetime.datetime.today())
    metadata['kernels'] = str(kernel_variation)
    with open('./models/{}-{}-{}trained-with-adversarial-examples-{}-{}.metadata'.format(dataset_type.value, encoding_type.value, cache_prefix, str(kernel_variation), adversarial_number), 'w') as file:
        json.dump(metadata, file, indent=4,)

    model.save('./models/{}-{}-{}trained-with-adversarial-examples-{}-{}.h5'.format(dataset_type.value, encoding_type.value, cache_prefix, str(kernel_variation), adversarial_number))

    # Return model
    return model


if __name__ == '__main__':
    for dataset_type in dataset.DatasetType:
        for adversarial_number in [-1, 500, 1000, 1500, 2000]:
            print('Dataset: {}, Adversarials: {}'.format(dataset_type.value, adversarial_number))
            model = get_network(dataset_type, dataset.Encoding.WORD2VEC, adversarial_number=adversarial_number)

# kate: replace-tabs true; indent-width 4; indent-mode python;
