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

import enum
import urllib.request
import os
import logging
import shutil
import re
import numpy
import bz2
import xml
import json
import gzip
import tempfile

import helpers


@enum.unique
class DatasetType(enum.Enum):
    """
    Enum representing the different datasets used for training
    """
    TREC = 'trec'
    AG = 'ag'
    AMAZONMOVIE = 'amazonmovie'


@enum.unique
class Encoding(enum.Enum):
    """
    Enum representing the different encodings for text implemented
    """
    WORD2VEC = 'word2vec'
    CHARACTER = 'character'
    NONE = 'none'


def get_standard_dataset(dataset_type: DatasetType, encoding_type: Encoding):
    """
    Returns selected dataset with standard options
    :param dataset_type: Dataset selection
    :type dataset_type: DatasetType
    :param encoding_type: Encoding selection
    :type encoding_type: Encoding
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    return {DatasetType.TREC.value: get_trec_dataset,
            DatasetType.AG.value: get_ag_dataset,
            DatasetType.AMAZONMOVIE.value: get_amazon_movie_dataset,
            }[dataset_type.value](encoding=encoding_type)


def _encode_none(train, test):
    """
    Returns both arrays not encoded

    The purpose for this encoding form is to get the strings unencoded for debugging purpose. This could also be used to implement a highly customised representation on the original string representation.

    :param train: Trainings data
    :type train: array
    :param test: Test data
    :type test: array
    :returns: train, test as (not) encoded numpy arrays
    """
    print('Encoding None')
    return numpy.array(train), numpy.array(test)


def _encode_word2vec(train, test):
    """
    Returns both inumpyut arrays in Google word2vec representation.

    Both arrays must have the form [str(), str(), ...]

    :param train: Trainings data
    :type train: array
    :param test: Test data
    :type test: array
    :returns: train, test as encoded numpy arrays
    """
    w2v = helpers.get_w2v()

    # Get length:
    max_length = 0
    for data in train:
        max_length = max(max_length, len(data.split(" ")))
    for data in test:
        max_length = max(max_length, len(data.split(" ")))

    encoded_train = numpy.memmap(tempfile.TemporaryFile(), dtype='float', mode='w+', shape=(len(train), max_length, 300))
    encoded_test = numpy.memmap(tempfile.TemporaryFile(), dtype='float', mode='w+', shape=(len(test), max_length, 300))

    def _encode_one_array(input_array, output_array):
        data_length = len(input_array)
        for i_data in range(data_length):
            print('Encoding word2vec - {0:5d} of {1:5d}'.format(i_data+1, data_length), end='\r')
            data = input_array[i_data].split(" ")
            for i_word in range(len(data)):
                encoded = None
                word_found = False
                if data[i_word] in w2v:
                    encoded = w2v[data[i_word]]
                    word_found = True
                elif data[i_word].lower() in w2v:
                    encoded = w2v[data[i_word].lower()]
                    word_found = True

                if word_found:
                    for i in range(len(encoded)):
                        output_array[i_data][i_word][i] = encoded[i]

    print('Encoding train set')
    _encode_one_array(train, encoded_train)
    print()
    print('Encoding test set')
    _encode_one_array(test, encoded_test)
    print()
    return encoded_train, encoded_test



def _encode_character(train, test):
    """
    Returns both arrays in character encoding.

    Character encoding is a one-hot encoding where each character in the sentence gets encoded as an array.
    Example:

    f = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    :param train: Trainings data
    :type train: array
    :param test: Test data
    :type test: array
    :returns: train, test as encoded numpy arrays
    """
    (character_index_dict, number_indices) = helpers.get_character_index_dict()
    # Get length:
    max_length = 0
    for data in train:
        max_length = max(max_length, len(data))
    for data in test:
        max_length = max(max_length, len(data))

    encoded_train = numpy.memmap(tempfile.TemporaryFile(), dtype='float', mode='w+', shape=(len(train), max_length, number_indices))
    encoded_test = numpy.memmap(tempfile.TemporaryFile(), dtype='float', mode='w+', shape=(len(test), max_length, number_indices))


    def _encode_one_array(input_array, output_array):
        data_length = len(input_array)
        for i_data in range(data_length):
            print('Encoding character - {0:5d} of {1:5d}'.format(i_data+1, data_length), end='\r')
            for i_char in range(len(input_array[i_data])):
                encoded_character = None
                char_found = False
                try:
                    char_index = character_index_dict[input_array[i_data][i_char]]
                    output_array[i_data][i_char][char_index] = 1
                except KeyError:
                    pass

    print('Encoding train set')
    _encode_one_array(train, encoded_train)
    print()
    print('Encoding test set')
    _encode_one_array(test, encoded_test)
    print()

    return encoded_train, encoded_test


def get_trec_dataset(encoding=Encoding.WORD2VEC, detailed_classes=False):
    """
    Returns the TREC dataset with the specified encoding

    If the dataset can not be found it will be downloaded. The labels start with 0 and are equal to class_labels[label].

    :param encoding: Encoding for the string representation
    :type encoding: Encoding
    :param detailed_classes: If set to True, the detailed classes will be used. Else only the overall topic will be used
    :type detailed_classes: bool
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    # Create dirs
    print('Getting TREC dataset')
    if not os.path.exists('./datasets/'):
        os.makedirs('./datasets/')
    if not os.path.exists('./datasets/TREC/'):
        os.makedirs('./datasets/TREC/')

    # Download dataset
    if not os.path.exists('./datasets/TREC/train_5500.label'):
        logging.debug('Downloading TREC: train_5500.label')
        urllib.request.urlretrieve('http://cogcomp.cs.illinois.edu/Data/QA/QC/train_5500.label', './datasets/TREC/train_5500.label')

    if not os.path.exists('./datasets/TREC/TREC_10.label'):
        logging.debug('Downloading TREC: TREC_10.label')
        urllib.request.urlretrieve('http://cogcomp.cs.illinois.edu/Data/QA/QC/TREC_10.label', './datasets/TREC/TREC_10.label')

    try:
        encoding_function = {
            Encoding.WORD2VEC: _encode_word2vec,
            Encoding.CHARACTER: _encode_character,
            Encoding.NONE: _encode_none,
        }[encoding]
    except KeyError:
        logging.error('Unknown encoding {}'.format(encoding))
        return (None, None), (None, None)

    class_labels = []
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    # Read training dataset
    with open('./datasets/TREC/train_5500.label', 'r', encoding='latin_1') as trainings_data:
        for line in trainings_data:
            line = line.split(' ', maxsplit=1)

            current_y = line[0]
            current_x = line[1]

            if not detailed_classes:
                current_y = current_y.split(':')[0]

            # Remove new line
            current_x = current_x.replace('\n', '')

            # Encode y
            try:
                current_y = class_labels.index(current_y)
            except ValueError:
                # Not in list
                class_labels.append(current_y)
                current_y = class_labels.index(current_y)

            # Add to dataset
            train_x.append(current_x)
            train_y.append(current_y)

    # Read test dataset
    with open('./datasets/TREC/TREC_10.label', 'r', encoding='latin_1') as test_data:
        for line in test_data:
            line = line.split(' ', maxsplit=1)

            current_y = line[0]
            current_x = line[1]

            if not detailed_classes:
                current_y = current_y.split(':')[0]

            # Remove new line
            current_x = current_x.replace('\n', '')

            # Encode y
            try:
                current_y = class_labels.index(current_y)
            except ValueError:
                # Not in list
                class_labels.append(current_y)
                current_y = class_labels.index(current_y)

            # Add to dataset
            test_x.append(current_x)
            test_y.append(current_y)

    train_x, test_x = encoding_function(train_x, test_x)

    return (train_x, train_y), (test_x, test_y), class_labels


def get_ag_dataset(encoding=Encoding.WORD2VEC, number_train_per_class=4000, number_test_per_class=400, filter_classes=['World', 'Entertainment', 'Sports', 'Business']):
    """
    Returns the AG's news dataset with the specified encoding. The input will consist of the descriptions of the articles.

    If the dataset can not be found it will be downloaded. The labels start with 0 and are equal to class_labels[label].
    The dataset is quite large and needs a lot of RAM for encoding, therefore an option to limit the number of entries is provided.

    :param encoding: Encoding for the string representation
    :type encoding: Encoding
    :param number_train_per_class: Number of data points for training per category. Should be smaller than number of entries of the category in the dataset.
    :type number_train_per_class: int
    :param number_test_per_class: Number of data points for testing per category.
    :type number_test_per_class: int
    :param filter_classes: Categories which should be included in the dataset. Per default all Categories with more than 10.000 entries are included (excepi Italia.
    :type filter_classes: list
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    print('Getting AG dataset')
    if not os.path.exists('./datasets/'):
        os.makedirs('./datasets/')
    if not os.path.exists('./datasets/AG/'):
        os.makedirs('./datasets/AG/')

    # Download dataset
    if not os.path.exists('./datasets/AG/newsspace200.xml.bz'):
        logging.debug('Downloading AG: newsspace200.xml.bz')
        urllib.request.urlretrieve('https://www.di.unipi.it/~gulli/newsspace200.xml.bz', './datasets/AG/newsspace200.xml.bz')

    xml_file = bz2.BZ2File('./datasets/AG/newsspace200.xml.bz', 'r')
    element_tree = xml.etree.ElementTree.parse(xml_file)
    root = element_tree.getroot()

    train_count = {x: 0 for x in filter_classes}
    test_count = {x: 0 for x in filter_classes}

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    class_labels = [x for x in filter_classes]

    text = None
    category = None

    for outer in root.iter():
        for element in outer.iter():
            if element.tag == 'source':
                # Flush current saved data
                if text is not None and category is not None:
                    # Clean data
                    if category in filter_classes:
                        text = ' '.join(text.split())  # Clean whitespace
                        if train_count[category] < number_train_per_class:
                            train_x.append(text)
                            train_y.append(class_labels.index(category))
                            train_count[category] = train_count[category] + 1
                        elif test_count[category] < number_test_per_class:
                            test_x.append(text)
                            test_y.append(class_labels.index(category))
                            test_count[category] = test_count[category] + 1
                    text = None
                    category = None
            elif element.tag == 'description':
                if element.text is not None:
                    if text is None:
                        text = ''
                    else:
                        text = text + ' '
                    text = text + element.text
            elif element.tag == 'category':
                category = element.text

    del root
    del element_tree
    del xml_file

    try:
        encoding_function = {
            Encoding.WORD2VEC: _encode_word2vec,
            Encoding.CHARACTER: _encode_character,
            Encoding.NONE: _encode_none,
        }[encoding]
    except KeyError:
        logging.error('Unknown encoding {}'.format(encoding))
        return (None, None), (None, None)

    train_x, test_x = encoding_function(train_x, test_x)

    return (train_x, train_y), (test_x, test_y), class_labels


def get_amazon_movie_dataset(encoding=Encoding.WORD2VEC, number_train_per_class=2000, number_test_per_class=200):
    """
    Returns the Amazon movie review dataset with the specified encoding. The input will consist of the descriptions of the articles.

    If the dataset can not be found it will be downloaded. The labels represent 0=bad (< 3.0) and 1=good (>= 3.0)
    The dataset is quite large and needs a lot of RAM for encoding, therefore an option to limit the number of entries is provided.

    :param encoding: Encoding for the string representation
    :type encoding: Encoding
    :param number_train_per_class: Number of data points for training per category. Should be smaller than number of entries of the category in the dataset.
    :type number_train_per_class: int
    :param number_test_per_class: Number of data points for testing per category.
    :type number_test_per_class: int
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    print('Getting Amazon Movie dataset')
    if not os.path.exists('./datasets/'):
        os.makedirs('./datasets/')
    if not os.path.exists('./datasets/AmazonMovie/'):
        os.makedirs('./datasets/AmazonMovie/')

    # Download dataset
    if not os.path.exists('./datasets/AmazonMovie/movies.txt.gz'):
        logging.debug('Downloading AmazonMovie: movies.txt.gz')
        urllib.request.urlretrieve('http://snap.stanford.edu/data/movies.txt.gz', './datasets/AmazonMovie/movies.txt.gz')

    class_labels = ['bad (<= 2.0)', 'good (>= 4.0)']

    train_count = [0, 0]
    test_count = [0, 0]

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    score = -1.0
    text = ''

    with gzip.open('./datasets/AmazonMovie/movies.txt.gz', 'r') as file:
        for line in file:
            line = line.decode('iso-8859-1')
            line = line.split(':')
            if len(line) >= 2:
                line[1] = line[1].strip()
            if len(line) != 2:  # Assume the record is over
                if text != '' and score !=  -1.0:
                    category = 0
                    if score >= 4.0:
                        category = 1
                    elif score > 2.0:
                        continue
                    if train_count[category] < number_train_per_class:
                        train_x.append(text)
                        train_y.append(category)
                        train_count[category] = train_count[category] + 1
                    elif test_count[category] < number_test_per_class:
                        test_x.append(text)
                        test_y.append(category)
                        test_count[category] = test_count[category] + 1
                        if test_count[0] == number_test_per_class and test_count[1] == number_test_per_class:
                            break  # Early way out when we don't need more data
                text = ''
                score = -1.0
            elif line[0] == 'review/score':
                try:
                    score = float(line[1])
                except ValueError:
                    logging.debug('Can not convert ' + line[1] + ' to float')
            elif line[0] == 'review/text' or line[0] == 'review/summary':
                text += line[1] + ' '

    try:
        encoding_function = {
            Encoding.WORD2VEC: _encode_word2vec,
            Encoding.CHARACTER: _encode_character,
            Encoding.NONE: _encode_none,
        }[encoding]
    except KeyError:
        logging.error('Unknown encoding {}'.format(encoding))
        return (None, None), (None, None)

    train_x, test_x = encoding_function(train_x, test_x)

    return (train_x, train_y), (test_x, test_y), class_labels

def cache_dataset(x_train, y_train, x_test, y_test, class_labels, key):
    """
    Caches the given dataset on the hard drive.

    This makes it possible to retrieve the dataset later without having to go through encoding etc. later.
    Datasets will be saved in /cache/dataset/$KEY/

    :param x_train: Train data
    :type x_train: numpy.array or similar
    :param y_train: Train label
    :type y_train: numpy.array or similar
    :param x_test: test data
    :type x_test: numpy.array or similar
    :param y_test: test label
    :type y_test: numpy.array or similar
    :param class_labels: Array containing class labels
    :type class_labels: array
    :param key: Key for saving / retrieving
    :type key: str
    """
    if not os.path.exists('./cache/'):
        os.makedirs('./cache/')
    if not os.path.exists('./cache/dataset/'):
        os.makedirs('./cache/dataset/')
    if not os.path.exists('./cache/dataset/{}/'.format(key)):
        os.makedirs('./cache/dataset/{}/'.format(key))

    numpy.save('./cache/dataset/{}/x_train.npy'.format(key), x_train)
    numpy.save('./cache/dataset/{}/y_train.npy'.format(key), y_train)
    numpy.save('./cache/dataset/{}/x_test.npy'.format(key), x_test)
    numpy.save('./cache/dataset/{}/y_test.npy'.format(key), y_test)
    with open('./cache/dataset/{}/class_labels.json'.format(key), 'w') as file:
        json.dump(class_labels, file)


def get_from_cache(key, memmap=None):
    """
    Retrieves cached dataset

    :raise IOError: Cached dataset does not exist
    :param key: Key of the dataset
    :type key: str
    :param memmap: If set to None, dataset will be loaded into cache. Otherwise dataset will be memmapped using the given mode
    :type memmap: {None, ‘r+’, ‘r’, ‘w+’, ‘c’}
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    if not os.path.exists('./cache/'):
        raise IOError('Cache not existing')
    if not os.path.exists('./cache/dataset/'):
        raise IOError('Dataset cache not existing')
    if not os.path.exists('./cache/dataset/{}/'.format(key)):
        raise IOError('Dataset key not existing')

    x_train = numpy.load('./cache/dataset/{}/x_train.npy'.format(key), mmap_mode=memmap)
    y_train = numpy.load('./cache/dataset/{}/y_train.npy'.format(key), mmap_mode=memmap)
    x_test = numpy.load('./cache/dataset/{}/x_test.npy'.format(key), mmap_mode=memmap)
    y_test = numpy.load('./cache/dataset/{}/y_test.npy'.format(key), mmap_mode=memmap)
    with open('./cache/dataset/{}/class_labels.json'.format(key), 'r') as file:
        class_labels = json.load(file)

    return (x_train, y_train), (x_test, y_test), class_labels


def is_cached(key):
    """
    Checks if the given cached dataset exists

    :param key: Key of the dataset
    :type key: str
    :returns: True if dataset exists, False otherwise
    """
    return os.path.exists('./cache/dataset/{}/'.format(key))


if __name__ == '__main__':
    # Create cache
    for encoding_type in Encoding:
        for dataset_type in DatasetType:
            if not is_cached('{}-{}'.format(dataset_type.value, encoding_type.value)):
                print('Creating cache for {} ({} encoding)'.format(dataset_type.value, encoding_type.value))
                print("Key: '{}-{}'".format(dataset_type.value, encoding_type.value))
                (x_train, y_train), (x_test, y_test), class_labels = get_standard_dataset(dataset_type, encoding_type)
                cache_dataset(x_train, y_train, x_test, y_test, class_labels, '{}-{}'.format(dataset_type.value, encoding_type.value))
                del x_train, y_train, x_test, y_test, class_labels

# kate: replace-tabs true; indent-width 4; indent-mode python;
