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
import helpers


_WORD2VEC_CACHE = None


def encode_character(sentence, min_length=0):
    """
    Encodes a single sentence into character encoding.

    Character encoding is a one-hot encoding where each character in the sentence gets encoded as an array.
    Example:

    f = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,...]

    :param sentence: Sentence to encode
    :type sentence: str
    :param min_length: minimum length the sentence must have. If the sentence is shorter zero arrays will be added.
    :type min_length: int
    :returns: numpy.array containing encoded sentence
    """
    encoded = []
    (character_index_dict, number_indices) = helpers.get_character_index_dict()
    for charachter in sentence:
        try:
            char_index = character_index_dict[charachter]
            encoded_character = numpy.zeros(number_indices)
            encoded_character[char_index] = 1
            encoded.append(encoded_character)
        except KeyError:
            encoded.append(numpy.zeros(number_indices))
    while len(encoded) < min_length:
        encoded.append(numpy.zeros(number_indices))
    return numpy.array(encoded)


def encode_word2vec(sentence, min_length=0, cache_model=False):
    """
    Returns the given sentence as word2vec encoding.

    PLEASE NOTE: Calling this method takes huge amounts of ram due to the large model (5+GB).
    If cache_model is set to True, this memory will be occupied until unload_word2vec is called. However, the method call will be much faster due to not needing to load the model from disk again.

    :param sentence: Sentence to encode
    :type sentence: str
    :param min_length: minimum length the sentence must have. If the sentence is shorter zero arrays will be added.
    :type min_length: int
    :param cache_model: If set to True, the word2vec model will be cached in RAM
    :type cache_model: bool
    :returns: numpy.array containing encoded sentence
    """
    # Load w2v
    w2v = None
    global _WORD2VEC_CACHE
    if _WORD2VEC_CACHE is not None:
        # Use cached
        w2v = _WORD2VEC_CACHE
    else:
        # Load model
        w2v = helpers.get_w2v()
        if cache_model:
            _WORD2VEC_CACHE = w2v
    encoded = []
    sentence = sentence.split(" ")
    for word in sentence:
        if word in w2v:
            encoded.append(w2v[word])
        elif word.lower() in w2v:
            encoded.append(w2v[word.lower()])
        else:
            encoded.append(numpy.zeros(300))
    while len(encoded) < min_length:
        encoded.append(numpy.zeros(300))
    return numpy.array(encoded)


def load_word2vec():
    """
    Loads the word2vec model into cache.

    PLEASE NOTE: Calling this method takes huge amounts of ram due to the large model (5+GB).
    This ram will be occupied until the model is unload (unload_word2vec is called).
    """
    global _WORD2VEC_CACHE
    if _WORD2VEC_CACHE is None:
        _WORD2VEC_CACHE = helpers.get_w2v()


def unload_word2vec():
    """
    Unloads the cached word2vec model.

    If the model is cached, this may free up to 5GB or more RAM.

    If the model is not cached, this should have no effect.
    """
    _WORD2VEC_CACHE = None


if __name__ == '__main__':
    print("Encoding 'cake'")
    print()
    print('Character encoding')
    print(encode_character('cake'))
    print()
    print('Word2vec encoding')
    print(encode_word2vec('cake'))

# kate: replace-tabs true; indent-width 4; indent-mode python;
