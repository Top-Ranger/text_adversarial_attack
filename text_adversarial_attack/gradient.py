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

import numpy
import keras
import keras.backend
import tensorflow


def get_gradient(network: keras.models.Sequential, X: numpy.array, session: tensorflow.InteractiveSession) -> numpy.array:
    """
    Calculates the gradient for a given network/input
    :param network: Network
    :type network: keras.Sequential
    :param X: Input to network.
    :type X: numpy.array
    :param session: Session object
    :type session: tensorflow.InteractiveSession
    :returns: Gradient as numpy.array
    """
    x = tensorflow.placeholder(tensorflow.float32, X.shape)

    prediction = network(x)

    y_shape = tensorflow.shape(prediction)
    classes = y_shape[1]
    index = tensorflow.argmax(prediction, axis=1)
    target = tensorflow.one_hot(index, classes, on_value=1.0, off_value=0.0)

    logits, = prediction.op.inputs
    loss = tensorflow.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits)
    gradient, = tensorflow.gradients(loss, x)

    return session.run(gradient, feed_dict={x: X, keras.backend.learning_phase(): 0})

# kate: replace-tabs true; indent-width 4; indent-mode python;
