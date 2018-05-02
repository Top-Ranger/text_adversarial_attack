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
import random

import helpers
import dataset


class NoAdversarialData(Exception):
    """
    This exception is risen when no adversarial data is found.
    The adversarial data is expected to be in the folder './adversarial_example'
    """
    pass

class NotEnoughAdversarialData(Exception):
    """
    This exception is risen when adversarial data is found, however not enough examples are generated for the requested dataset.
    """
    pass


def _load_adversarial_data(dataset_type: dataset.DatasetType) -> list:
    """
    Loads the adversarial examples for a given dataset

    The order of the adversarial examples is randomised

    :raises: NoAdversarialData

    :param dataset_type: Target dataset
    :type dataset_type: dataset.Dataset
    :return: list of adversarial examples, list of input classes
    """
    if not os.path.exists('./adversarial_example/'):
        raise NoAdversarialData("Dataset {} - Folder './adversarial_example/' not found (no adversarial data created?)".format(dataset_type.value))

    adversarial_data = []
    input_classes = []

    # Random the order to counter effects of the order of examples
    file_list = os.listdir('./adversarial_example/')
    random.SystemRandom().shuffle(file_list)  # SystemRandom for enough states to create all permutations

    for file in file_list:
        if not file.endswith('.json'):
            continue
        result = dict()
        with open('./adversarial_example/{}'.format(file)) as result_file:
            result = json.load(result_file)
        if 'dataset' in result and 'adversarial_example' in result and 'input_class' in result and result['dataset'] == dataset_type.value:
            adversarial_data.append(result['adversarial_example'])
            input_classes.append(result['input_class'])

    if len(adversarial_data) == 0:
        raise NoAdversarialData("Dataset {} - Folder './adversarial_example/' found, but it containes no adversarial examples".format(dataset_type.value))

    return adversarial_data, input_classes


def get_standard_dataset(dataset_type: dataset.DatasetType, encoding_type: dataset.Encoding, adversarial_number=-1):
    """
    Returns selected dataset with standard options

    :raises: NoAdversarialData
    :raises: NotEnoughAdversarialData

    :param dataset_type: Dataset selection
    :type dataset_type: DatasetType
    :param encoding_type: Encoding selection
    :type encoding_type: Encoding
    :param adversarial_number: Number of adversarial data added. Set -1 to add all data
    :type adversarial_number: int
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    return {dataset.DatasetType.TREC.value: get_trec_dataset,
            dataset.DatasetType.AG.value: get_ag_dataset,
            dataset.DatasetType.AMAZONMOVIE.value: get_amazon_movie_dataset,
            }[dataset_type.value](encoding=encoding_type, adversarial_number=adversarial_number)


def get_trec_dataset(encoding=dataset.Encoding.WORD2VEC, detailed_classes=False, adversarial_number=-1):
    """
    Returns the TREC dataset with the specified encoding

    If the dataset can not be found it will be downloaded. The labels start with 0 and are equal to class_labels[label].

    Please note that detailed_classes must be the same for the generation of adversarial classes as for getting this dataset, otherwise there will be a mismatch between class meaning.

    :raises: NoAdversarialData
    :raises: NotEnoughAdversarialData

    :param encoding: Encoding for the string representation
    :type encoding: Encoding
    :param detailed_classes: If set to True, the detailed classes will be used. Else only the overall topic will be used
    :type detailed_classes: bool
    :param adversarial_number: Number of adversarial data added. Set -1 to add all data
    :type adversarial_number: int
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    # Create dirs
    print('Getting adversarial TREC dataset ({} examples)'.format(adversarial_number))
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
            dataset.Encoding.WORD2VEC: dataset._encode_word2vec,
            dataset.Encoding.CHARACTER: dataset._encode_character,
            dataset.Encoding.NONE: dataset._encode_none,
        }[encoding]
    except KeyError:
        logging.error('Unknown encoding {}'.format(encoding))
        return (None, None), (None, None)

    class_labels = []
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    adversarial_x, adversarial_y = _load_adversarial_data(dataset.DatasetType.TREC)
    if adversarial_number == -1:
        train_x = adversarial_x
        train_y = adversarial_y
    else:
        if len(adversarial_x) < adversarial_number:
            raise NotEnoughAdversarialData('TREC: To few adversarial examples to add to the dataset. Requested: {}, Available: {}'.format(adversarial_number, len(adversarial_x)))
        train_x = adversarial_x[:adversarial_number]
        train_y = adversarial_y[:adversarial_number]

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


def get_ag_dataset(encoding=dataset.Encoding.WORD2VEC, number_train_per_class=4000, number_test_per_class=400, filter_classes=['World', 'Entertainment', 'Sports', 'Business'], adversarial_number=-1):
    """
    Returns the AG's news dataset with the specified encoding. The input will consist of the descriptions of the articles.

    If the dataset can not be found it will be downloaded. The labels start with 0 and are equal to class_labels[label].
    The dataset is quite large and needs a lot of RAM for encoding, therefore an option to limit the number of entries is provided.

    Please note that filter_classes must be the same for the generation of adversarial classes as for getting this dataset, otherwise there will be a mismatch between class meaning.

    :raises: NoAdversarialData
    :raises: NotEnoughAdversarialData

    :param encoding: Encoding for the string representation
    :type encoding: Encoding
    :param number_train_per_class: Number of data points for training per category. Should be smaller than number of entries of the category in the dataset.
    :type number_train_per_class: int
    :param number_test_per_class: Number of data points for testing per category.
    :type number_test_per_class: int
    :param filter_classes: Categories which should be included in the dataset. Per default all Categories with more than 10.000 entries are included (excepi Italia.
    :type filter_classes: list
    :param adversarial_number: Number of adversarial data added. Set -1 to add all data
    :type adversarial_number: int
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    print('Getting adversarial AG dataset ({} examples)'.format(adversarial_number))
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

    adversarial_x, adversarial_y = _load_adversarial_data(dataset.DatasetType.AG)
    if adversarial_number == -1:
        train_x = adversarial_x
        train_y = adversarial_y
    else:
        if len(adversarial_x) < adversarial_number:
            raise NotEnoughAdversarialData('AG: To few adversarial examples to add to the dataset. Requested: {}, Available: {}'.format(adversarial_number, len(adversarial_x)))
        train_x = adversarial_x[:adversarial_number]
        train_y = adversarial_y[:adversarial_number]

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
            dataset.Encoding.WORD2VEC: dataset._encode_word2vec,
            dataset.Encoding.CHARACTER: dataset._encode_character,
            dataset.Encoding.NONE: dataset._encode_none,
        }[encoding]
    except KeyError:
        logging.error('Unknown encoding {}'.format(encoding))
        return (None, None), (None, None)

    train_x, test_x = encoding_function(train_x, test_x)

    return (train_x, train_y), (test_x, test_y), class_labels


def get_amazon_movie_dataset(encoding=dataset.Encoding.WORD2VEC, number_train_per_class=2000, number_test_per_class=200, adversarial_number=-1):
    """
    Returns the Amazon movie review dataset with the specified encoding. The input will consist of the descriptions of the articles.

    If the dataset can not be found it will be downloaded. The labels represent 0=bad (< 3.0) and 1=good (>= 3.0)
    The dataset is quite large and needs a lot of RAM for encoding, therefore an option to limit the number of entries is provided.

    :raises: NoAdversarialData
    :raises: NotEnoughAdversarialData

    :param encoding: Encoding for the string representation
    :type encoding: Encoding
    :param number_train_per_class: Number of data points for training per category. Should be smaller than number of entries of the category in the dataset.
    :type number_train_per_class: int
    :param number_test_per_class: Number of data points for testing per category.
    :type number_test_per_class: int
    :param adversarial_number: Number of adversarial data added. Set -1 to add all data
    :type adversarial_number: int
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    print('Getting adversarial Amazon Movie dataset ({} examples)'.format(adversarial_number))
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

    adversarial_x, adversarial_y = _load_adversarial_data(dataset.DatasetType.AMAZONMOVIE)
    if adversarial_number == -1:
        train_x = adversarial_x
        train_y = adversarial_y
    else:
        if len(adversarial_x) < adversarial_number:
            raise NotEnoughAdversarialData('Amazon movie: To few adversarial examples to add to the dataset. Requested: {}, Available: {}'.format(adversarial_number, len(adversarial_x)))
        train_x = adversarial_x[:adversarial_number]
        train_y = adversarial_y[:adversarial_number]

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
            dataset.Encoding.WORD2VEC: dataset._encode_word2vec,
            dataset.Encoding.CHARACTER: dataset._encode_character,
            dataset.Encoding.NONE: dataset._encode_none,
        }[encoding]
    except KeyError:
        logging.error('Unknown encoding {}'.format(encoding))
        return (None, None), (None, None)

    train_x, test_x = encoding_function(train_x, test_x)

    return (train_x, train_y), (test_x, test_y), class_labels


# kate: replace-tabs true; indent-width 4; indent-mode python;
