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

import os
import random
import json
import numpy
import sys
import datetime
import time
import copy

import dataset
import encode_sentence
import trained_networks
import attacks

import tensorflow
import keras.backend


# Constants
INPUT_TRIES = 50
NUMBER_CREATED = 10

# Dataset choice
print()
print('--- Dataset choice ---')
DATASET = random.choice(list(dataset.DatasetType))
#DATASET = dataset.DatasetType.TREC
print('Chosen: {}'.format(DATASET.value))

# Start session
print()
print('--- Start session ---')
session = tensorflow.InteractiveSession()
keras.backend.set_session(session)


# Preparations
print()
print('--- Preperations ---')
if not os.path.exists('./adversarial_example/'):
    os.makedirs('./adversarial_example/')
if not os.path.exists('./adversarial_example_unsuccessful/'):
    os.makedirs('./adversarial_example_unsuccessful/')
if not os.path.exists('./adversarial_example_input_unusable/'):
    os.makedirs('./adversarial_example_input_unusable/')


print('Loading word2vec method into cache')
encode_sentence.load_word2vec()


# Get networks
print()
print('--- Get networks ---')
target_w2v = trained_networks.get_network(DATASET, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0)
target_w2v_retrained = trained_networks.get_network(DATASET, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0, cache_prefix='retrained-')
target_w2v_second_half = trained_networks.get_network(DATASET, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=1)
target_w2v_alternative = trained_networks.get_network(DATASET, dataset.Encoding.WORD2VEC, kernel_variation=[3, 3, 5, 5], every_xth_trainings_data=2, skip_trainings_data=0)
target_character = trained_networks.get_network(DATASET, dataset.Encoding.CHARACTER, every_xth_trainings_data=2, skip_trainings_data=0)


# Get datasets
print()
print('--- Get datasets ---')
if dataset.is_cached('{}-none'.format(DATASET.value)):
    print('Using cached dataset')
    (dataset_x_train, dataset_y_train), (dataset_x_test, dataset_y_test), trec_class_labels = dataset.get_from_cache('{}-none'.format(DATASET.value), memmap='r')
else:
    print('Creating dataset')
    (dataset_x_train, dataset_y_train), (dataset_x_test, dataset_y_test), trec_class_labels = dataset.get_standard_dataset(DATASET, dataset.Encoding.NONE)
    dataset.cache_dataset(dataset_x_train, dataset_y_train, dataset_x_test, dataset_y_test, trec_class_labels, '{}-none'.format(DATASET.value))


for round in range(NUMBER_CREATED):
    result_dict = dict()
    result_dict['time'] = str(datetime.datetime.today())
    result_dict['dataset'] = DATASET.value

    # Find suitable input
    # An input is suitable if it is classified correctly for all tested networks
    print()
    print('--- Find suitable input ({}/{}) ---'.format(round+1, NUMBER_CREATED))
    found_input = False
    input_tries = 1
    input_index = 0
    while not found_input and input_tries <= INPUT_TRIES:
        print('Try {}'.format(input_tries))
        input_tries = input_tries + 1

        input_index = random.randint(0, len(dataset_x_train))

        # Test if we have already tried this input
        if os.path.exists('./adversarial_example/{}-{}.json'.format(DATASET.value, input_index)) or os.path.exists('./adversarial_example_unsuccessful/{}-{}.json'.format(DATASET.value, input_index)) or os.path.exists('./adversarial_example_input_unusable/{}-{}.json'.format(DATASET.value, input_index)):
            print('(duplicate input)')
            continue

        input_target_w2v = numpy.asarray([encode_sentence.encode_word2vec(dataset_x_train[input_index], cache_model=True)])
        output_class_target_w2v = target_w2v.predict(input_target_w2v)
        output_class_target_w2v = numpy.argmax(output_class_target_w2v)

        output_class_target_w2v_retrained = target_w2v_retrained.predict(input_target_w2v)
        output_class_target_w2v_retrained = numpy.argmax(output_class_target_w2v_retrained)

        output_class_target_w2v_second_half = target_w2v_second_half.predict(input_target_w2v)
        output_class_target_w2v_second_half = numpy.argmax(output_class_target_w2v_second_half)

        output_class_target_w2v_alternative = target_w2v_alternative.predict(input_target_w2v)
        output_class_target_w2v_alternative = numpy.argmax(output_class_target_w2v_alternative)

        input_target_character = numpy.asarray([encode_sentence.encode_character(dataset_x_train[input_index])])
        output_class_target_character = target_character.predict(input_target_character)
        output_class_target_character = numpy.argmax(output_class_target_character)

        if output_class_target_w2v == dataset_y_train[input_index] and output_class_target_w2v_retrained == dataset_y_train[input_index] and output_class_target_w2v_second_half == dataset_y_train[input_index] and output_class_target_w2v_alternative == dataset_y_train[input_index] and output_class_target_character == dataset_y_train[input_index]:
            found_input = True
        else:
            # Save unusable input
            result_unusable_input = copy.deepcopy(result_dict)
            result_unusable_input['input_sentence'] = dataset_x_train[input_index]
            result_unusable_input['input_index'] = input_index
            result_unusable_input['input_class'] = int(dataset_y_train[input_index])
            result_unusable_input['networks'] = [['w2v_target', int(output_class_target_w2v), bool(output_class_target_w2v == dataset_y_train[input_index])],
                                                 ['w2v_retrained', int(output_class_target_w2v_retrained), bool(output_class_target_w2v_retrained == dataset_y_train[input_index])],
                                                 ['w2v_second_half', int(output_class_target_w2v_second_half), bool(output_class_target_w2v_second_half == dataset_y_train[input_index])],
                                                 ['w2v_alternative_first_half', int(output_class_target_w2v_alternative), bool(output_class_target_w2v_alternative == dataset_y_train[input_index])],
                                                 ['character', int(output_class_target_character), bool(output_class_target_character == dataset_y_train[input_index])],
                                                 ]
            with open('./adversarial_example_input_unusable/{}-{}.json'.format(DATASET.value, input_index), 'w') as file:
                json.dump(result_unusable_input, file, indent=4,)

    if not found_input:
        print('No input found - exiting')
        continue

    result_dict['input_sentence'] = dataset_x_train[input_index]
    result_dict['input_index'] = input_index
    result_dict['input_class'] = int(dataset_y_train[input_index])

    print('Using training sentence {}: {}'.format(input_index, dataset_x_train[input_index]))

    # Try finding adversarial example
    print()
    print('--- Try finding adversarial example ({}/{}) ---'.format(round+1, NUMBER_CREATED))

    running_time = time.process_time()

    result_dict['targetted_network'] = 'w2v_first_half'
    try:
        found_adversarial, number_changes, changes_list = attacks.attack_w2v(target_w2v, dataset_x_train[input_index], session, dataset_x_train, dataset_y_train, attack_key='{}_w2v_attack'.format(DATASET.value), use_keywords=True)
        found_adversarial_w2v = numpy.asarray([encode_sentence.encode_word2vec(found_adversarial, cache_model=True)])
    except attacks.NoAdversarialExampleFound:
        result_dict['successful'] = False
        with open('./adversarial_example_unsuccessful/{}-{}.json'.format(DATASET.value, input_index), 'w') as file:
            json.dump(result_dict, file, indent=4,)
        print()
        print('--- No adversarial example found, aborting ---')
        continue

    adversarial_class = int(numpy.argmax(target_w2v.predict(found_adversarial_w2v)))

    running_time = time.process_time() - running_time

    result_dict['successful'] = True
    result_dict['adversarial_example'] = found_adversarial
    result_dict['adversarial_example_class'] = adversarial_class
    result_dict['running_time'] = running_time
    result_dict['number_changes'] = number_changes
    result_dict['changes'] = changes_list

    # Transferability
    print()
    print('--- Transferability ({}/{}) ---'.format(round+1, NUMBER_CREATED))

    transferability_list = list()

    # Retrained network
    target_w2v_retrained_adversarial_class = int(numpy.argmax(target_w2v_retrained.predict(found_adversarial_w2v)))
    transferability_list.append(['w2v_retrained', int(dataset_y_train[input_index]) != target_w2v_retrained_adversarial_class, target_w2v_retrained_adversarial_class])

    # w2v second half network
    target_w2v_second_half_adversarial_class = int(numpy.argmax(target_w2v_second_half.predict(found_adversarial_w2v)))
    transferability_list.append(['w2v_second_half', int(dataset_y_train[input_index]) != target_w2v_second_half_adversarial_class, target_w2v_second_half_adversarial_class])

    # w2v alternative network
    target_w2v_alternative_adversarial_class = int(numpy.argmax(target_w2v_alternative.predict(found_adversarial_w2v)))
    transferability_list.append(['w2v_alternative_first_half', int(dataset_y_train[input_index]) != target_w2v_alternative_adversarial_class, target_w2v_alternative_adversarial_class])

    # character encoding
    found_adversarial_character = numpy.asarray([encode_sentence.encode_character(found_adversarial)])
    target_character_adversarial_class = int(numpy.argmax(target_character.predict(found_adversarial_character)))
    transferability_list.append(['character', int(dataset_y_train[input_index]) != target_character_adversarial_class, target_character_adversarial_class])

    # Save results
    result_dict['transferability'] = transferability_list  # Network, transferability?, result class

    # Save results
    print()
    print('--- Save results ({}/{}) ---'.format(round+1, NUMBER_CREATED))
    with open('./adversarial_example/{}-{}.json'.format(DATASET.value, input_index), 'w') as file:
        json.dump(result_dict, file, indent=4,)


# Finishing
print()
print('--- Finishing ---')

encode_sentence.unload_word2vec()

print()
print('--- Done ---')
print()

# kate: replace-tabs true; indent-width 4; indent-mode python;
