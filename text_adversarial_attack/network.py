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

import keras
import keras.models
import keras.layers
import keras.layers.pooling


def get_single_layer_network(classes, representation_size, kernels=[3, 4, 5]):
    """
    Returns a network with a single convolutional layer with different kernel sizes.

    Network is not compiled.
    :param classes: Number of output classes
    :type classes: int
    :param representation_size: Length of representation vector, e.g. 300 for Google word2vec pretrained model
    :type representation_size: int
    :param kernels: Kernel sizes. Each entry equals to 25 kernels
    :type kernels: array of ints
    :returns: keras.models.Sequential
    """
    input_tensor = keras.layers.Input(shape=(None, representation_size))
    kernel_list = []
    for kernel in kernels:
        single_kernel = keras.models.Sequential()
        single_kernel = keras.layers.Convolution1D(filters=25, kernel_size=kernel, activation='relu')(input_tensor)
        single_kernel = keras.layers.pooling.GlobalMaxPooling1D()(single_kernel)
        kernel_list.append(single_kernel)

    output_tensor = keras.layers.Concatenate()(kernel_list)
    convolutional_filters = keras.models.Model(outputs=output_tensor, inputs=input_tensor)

    network = keras.models.Sequential()
    network.add(convolutional_filters)
    network.add(keras.layers.Dropout(0.5))
    network.add(keras.layers.Dense(classes))
    network.add(keras.layers.Activation('softmax'))

    return network

# kate: replace-tabs true; indent-width 4; indent-mode python;
