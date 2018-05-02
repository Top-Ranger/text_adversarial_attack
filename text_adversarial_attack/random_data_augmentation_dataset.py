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

import numpy.random
import nltk
import nltk.corpus


# Cache nltk data set tests
_RANDOM_DATA_AUGMENTATION_DATASET_AVERAGED_PERCEPTRON_TAGGER_CHECKED = False
_RANDOM_DATA_AUGMENTATION_DATASET_WORDNET_CHECKED = False

# Constants
_MAX_NUMBER_TRIES = 500


def _get_augmented_data(x, y, number, p=0.5):
    """
    Returns a number of augmented data samples.

    :param x: Input samples
    :type x: list
    :param y: Input classes
    :type y: list
    :param number: Number of samples created
    :type number: int
    :param p: Probability used in geometric probability function
    :type p: float
    :return: List with generated data samples, list with classes
    """
    if number < 0:
        raise ValueError('Number of data must be 0 or greater')

    if number == 0:
        # Early jump out
        return [], []
    # Basic checks
    assert len(x) == len(y)

    number_changes = numpy.random.geometric(p=p, size=number)
    augmented_x = []
    augmented_y = []

    # Check for nltk datasets
    nltk.data.path = ['./datasets/nltk/']
    global _RANDOM_DATA_AUGMENTATION_DATASET_WORDNET_CHECKED
    global _RANDOM_DATA_AUGMENTATION_DATASET_AVERAGED_PERCEPTRON_TAGGER_CHECKED
    if not _RANDOM_DATA_AUGMENTATION_DATASET_WORDNET_CHECKED:
        nltk.download('wordnet', download_dir='./datasets/nltk/')
        _RANDOM_DATA_AUGMENTATION_DATASET_WORDNET_CHECKED = True
    if not _RANDOM_DATA_AUGMENTATION_DATASET_AVERAGED_PERCEPTRON_TAGGER_CHECKED:
        nltk.download('averaged_perceptron_tagger', download_dir='./datasets/nltk/')
        _RANDOM_DATA_AUGMENTATION_DATASET_AVERAGED_PERCEPTRON_TAGGER_CHECKED = True

    print('Running data augmentation')
    run = 0
    for data in number_changes:
        print('{0:6d} of {1:6d}'.format(run+1, number), end='\r')
        run += 1
        successful_run = False # In case we weren't able to find a suitable augmented data redo it
        unsuccessful_counter = 0

        while not successful_run:
            successful_run = True
            unsuccessful_counter = 0
            index = random.randrange(0, len(x))
            current_sentence = x[index]
            current_class = y[index]
            number_changed = 0

            # Tag sentence
            # IMPORTANT: These two have to be kept in sync!
            # This is to reduce miss-tags later after modifying
            current_words = current_sentence.split(" ")
            for i in range(len(current_words)):
                if current_words[i] == '':  # Avoid empty words for tagging
                    current_words[i] = ' '
            current_words_tagged = nltk.pos_tag(current_words)
            for i in range(len(current_words)):
                if current_words[i] == ' ':  # Reconstruct original sentence
                    current_words[i] = ''
                    current_words_tagged[i] = ('', current_words_tagged[i][1])
            assert len(current_words) == len(current_words_tagged)


            while number_changed < data:
                # Test for break out
                if unsuccessful_counter > _MAX_NUMBER_TRIES:
                    break

                replacement_index = random.randrange(0, len(current_words))
                pos = None
                if 'NN' in current_words_tagged[replacement_index][1]:  # Noun
                    pos = nltk.corpus.wordnet.NOUN
                elif 'VB' in current_words_tagged[replacement_index][1]:  # Verb
                    pos = nltk.corpus.wordnet.VERB
                elif 'JJ' in current_words_tagged[replacement_index][1]:  # Adjective
                    pos = nltk.corpus.wordnet.ADJ
                elif 'RB' in current_words_tagged[replacement_index][1]:  # Adverb
                    pos = nltk.corpus.wordnet.ADV
                else:
                    # Can not find synonyms - skip this
                    unsuccessful_counter += 1
                    continue

                candidate_set = set()
                assert pos is not None
                for synonym_class in nltk.corpus.wordnet.synsets(current_words[replacement_index], pos=pos):
                    for synonym in synonym_class.lemma_names():
                        candidate_set.add(str(synonym))

                if len(candidate_set) is 0:
                    # no candidates found
                    unsuccessful_counter += 1
                    continue

                replacement_word = random.sample(candidate_set, 1)[0]

                if replacement_word == current_words[replacement_index]:
                    continue

                current_words.pop(replacement_index)
                old_pos_class = current_words_tagged.pop(replacement_index)[1]
                for single_word in reversed(replacement_word.split('_')):  # Multiple words are connected by '_', e.g. 'a_lot'
                    current_words.insert(replacement_index, single_word)
                    current_words_tagged.insert(replacement_index, (single_word, old_pos_class))
                number_changed += 1

        augmented_x.append(' '.join(current_words))
        augmented_y.append(current_class)

    print()
    return augmented_x, augmented_y


def get_standard_dataset(dataset_type: dataset.DatasetType, encoding_type: dataset.Encoding, number_augmented=0):
    """
    Returns selected dataset with standard options

    :raises: NoAdversarialData
    :raises: NotEnoughAdversarialData

    :param dataset_type: Dataset selection
    :type dataset_type: DatasetType
    :param encoding_type: Encoding selection
    :type encoding_type: Encoding
    :param number_augmented: Number of augmented data added. Needs to be 0 or positive
    :type number_augmented: int
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    return {dataset.DatasetType.TREC.value: get_trec_dataset,
            dataset.DatasetType.AG.value: get_ag_dataset,
            dataset.DatasetType.AMAZONMOVIE.value: get_amazon_movie_dataset,
            }[dataset_type.value](encoding=encoding_type, number_augmented=number_augmented)


def get_trec_dataset(encoding=dataset.Encoding.WORD2VEC, detailed_classes=False, number_augmented=0):
    """
    Returns the TREC dataset with the specified encoding

    If the dataset can not be found it will be downloaded. The labels start with 0 and are equal to class_labels[label].

    Please note that detailed_classes must be the same for the generation of adversarial classes as for getting this dataset, otherwise there will be a mismatch between class meaning.

    :param encoding: Encoding for the string representation
    :type encoding: Encoding
    :param detailed_classes: If set to True, the detailed classes will be used. Else only the overall topic will be used
    :type detailed_classes: bool
    :param number_augmented: Number of augmented data added. Needs to be 0 or positive
    :type number_augmented: int
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    # Create dirs
    print('Getting randomly augmented TREC dataset ({} examples)'.format(number_augmented))
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

    # Append augmented data
    augmented_x, augmented_y = _get_augmented_data(train_x, train_y, number_augmented)
    train_x.extend(augmented_x)
    train_y.extend(augmented_y)

    train_x, test_x = encoding_function(train_x, test_x)

    return (train_x, train_y), (test_x, test_y), class_labels


def get_ag_dataset(encoding=dataset.Encoding.WORD2VEC, number_train_per_class=4000, number_test_per_class=400, filter_classes=['World', 'Entertainment', 'Sports', 'Business'], number_augmented=0):
    """
    Returns the AG's news dataset with the specified encoding. The input will consist of the descriptions of the articles.

    If the dataset can not be found it will be downloaded. The labels start with 0 and are equal to class_labels[label].
    The dataset is quite large and needs a lot of RAM for encoding, therefore an option to limit the number of entries is provided.

    Please note that filter_classes must be the same for the generation of adversarial classes as for getting this dataset, otherwise there will be a mismatch between class meaning.

    :param encoding: Encoding for the string representation
    :type encoding: Encoding
    :param number_train_per_class: Number of data points for training per category. Should be smaller than number of entries of the category in the dataset.
    :type number_train_per_class: int
    :param number_test_per_class: Number of data points for testing per category.
    :type number_test_per_class: int
    :param filter_classes: Categories which should be included in the dataset. Per default all Categories with more than 10.000 entries are included (excepi Italia.
    :type filter_classes: list
    :param number_augmented: Number of augmented data added. Needs to be 0 or positive
    :type number_augmented: int
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    print('Getting randomly augmented AG dataset ({} examples)'.format(number_augmented))
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

    # Append augmented data
    augmented_x, augmented_y = _get_augmented_data(train_x, train_y, number_augmented)
    train_x.extend(augmented_x)
    train_y.extend(augmented_y)

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


def get_amazon_movie_dataset(encoding=dataset.Encoding.WORD2VEC, number_train_per_class=2000, number_test_per_class=200, number_augmented=0):
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
    :param number_augmented: Number of augmented data added. Needs to be 0 or positive
    :type number_augmented: int
    :returns: (train_x, train_y), (test_x, test_y), class_labels as numpy.array
    """
    print('Getting randomly augmented Amazon Movie dataset ({} examples)'.format(number_augmented))
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
                if text != '' and score != -1.0:
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

    # Append augmented data
    augmented_x, augmented_y = _get_augmented_data(train_x, train_y, number_augmented)
    train_x.extend(augmented_x)
    train_y.extend(augmented_y)

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

