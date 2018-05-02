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
TEMPERATURE = 20

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
if not os.path.exists('./distillation_adversarial_example-{}/'.format(TEMPERATURE)):
    os.makedirs('./distillation_adversarial_example-{}/'.format(TEMPERATURE))
if not os.path.exists('./distillation_adversarial_example_unsuccessful-{}/'.format(TEMPERATURE)):
    os.makedirs('./distillation_adversarial_example_unsuccessful-{}/'.format(TEMPERATURE))
if not os.path.exists('./distillation_adversarial_example_input_unusable-{}/'.format(TEMPERATURE)):
    os.makedirs('./distillation_adversarial_example_input_unusable-{}/'.format(TEMPERATURE))


print('Loading word2vec method into cache')
encode_sentence.load_word2vec()


# Get networks
print()
print('--- Get networks ---')
target_w2v = trained_networks.get_defensively_distilled_network(DATASET, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0, temperature=TEMPERATURE)


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
        if os.path.exists('./distillation_adversarial_example-{}/{}-{}.json'.format(TEMPERATURE, DATASET.value, input_index)) or os.path.exists('./distillation_adversarial_example_unsuccessful-{}/{}-{}.json'.format(TEMPERATURE, DATASET.value, input_index)) or os.path.exists('./distillation_adversarial_example_input_unusable-{}/{}-{}.json'.format(TEMPERATURE, DATASET.value, input_index)):
            print('(duplicate input)')
            continue

        input_target_w2v = numpy.asarray([encode_sentence.encode_word2vec(dataset_x_train[input_index], cache_model=True)])
        output_class_target_w2v = target_w2v.predict(input_target_w2v)
        output_class_target_w2v = numpy.argmax(output_class_target_w2v)

        if output_class_target_w2v == dataset_y_train[input_index]:
            found_input = True
        else:
            # Save unusable input
            result_unusable_input = copy.deepcopy(result_dict)
            result_unusable_input['input_sentence'] = dataset_x_train[input_index]
            result_unusable_input['input_index'] = input_index
            result_unusable_input['input_class'] = int(dataset_y_train[input_index])
            result_unusable_input['temperature'] = TEMPERATURE
            result_unusable_input['networks'] = [['distillation_w2v_target', int(output_class_target_w2v), bool(output_class_target_w2v == dataset_y_train[input_index])],
                                                 ]
            with open('./distillation_adversarial_example_input_unusable-{}/{}-{}.json'.format(TEMPERATURE, DATASET.value, input_index), 'w') as file:
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

    result_dict['targetted_network'] = 'distillation_w2v'
    result_dict['temperature'] = TEMPERATURE
    try:
        found_adversarial, number_changes, changes_list = attacks.attack_w2v(target_w2v, dataset_x_train[input_index], session, dataset_x_train, dataset_y_train, attack_key='distillation-{}_{}_w2v_attack'.format(TEMPERATURE, DATASET.value), use_keywords=True)
        found_adversarial_w2v = numpy.asarray([encode_sentence.encode_word2vec(found_adversarial, cache_model=True)])
    except attacks.NoAdversarialExampleFound:
        result_dict['successful'] = False
        with open('./distillation_adversarial_example_unsuccessful-{}/{}-{}.json'.format(TEMPERATURE, DATASET.value, input_index), 'w') as file:
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

    # Save results
    print()
    print('--- Save results ({}/{}) ---'.format(round+1, NUMBER_CREATED))
    with open('./distillation_adversarial_example-{}/{}-{}.json'.format(TEMPERATURE, DATASET.value, input_index), 'w') as file:
        json.dump(result_dict, file, indent=4,)


# Finishing
print()
print('--- Finishing ---')

encode_sentence.unload_word2vec()

print()
print('--- Done ---')
print()

# kate: replace-tabs true; indent-width 4; indent-mode python;

