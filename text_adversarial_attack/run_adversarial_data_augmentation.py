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
import json
import tempfile

import trained_networks
import adversarial_trained_network
import adversarial_dataset
import dataset
import random_data_augmentation_trained_network

# Ensure tempfiles are created
if not os.path.exists('./cache/'):
    os.makedirs('./cache/')
tempfile.tempdir = './cache'

# Configuration
RUN_NAMES = ["test_1", "test_2", "test_3"]
MAX_NUMBER_DATA_AUGMENTATION = 8000
INCREASE_DATA_AUGMENTATION = 200

# Helpers
NETWORK_encoding_type = dataset.Encoding.WORD2VEC
NETWORK_kernel_variation = [3, 4, 5]
NETWORK_every_xth_trainings_data = 1
NETWORK_skip_trainings_data = 0

if not os.path.exists('./adevrsarial_data_augmentation/'):
    os.makedirs('./adevrsarial_data_augmentation/')

for run in RUN_NAMES:
    print('Starting run {}'.format(run))
    results = {'max_number_data_augmentation': MAX_NUMBER_DATA_AUGMENTATION,
               'increase_data_augmentation': INCREASE_DATA_AUGMENTATION,
               'run_name': run,
               'encoding': NETWORK_encoding_type.value
               }

    # Adversarial data augmentation
    for dataset_type in dataset.DatasetType:
        number_samples = []
        accuracies = []
        max_number_for_dataset = 0  # Ensure all data augmentations have the same range later
        for number_adversarial_data in range(0, MAX_NUMBER_DATA_AUGMENTATION+INCREASE_DATA_AUGMENTATION, INCREASE_DATA_AUGMENTATION):
            try:
                print('Adversarial Data Augmentation ({}): {}'.format(dataset_type.value, number_adversarial_data))
                network = adversarial_trained_network.get_network(dataset_type, NETWORK_encoding_type, number_adversarial_data, NETWORK_kernel_variation, cache_prefix=run)
                with open('./models/{}-{}-{}trained-with-adversarial-examples-{}-{}.metadata'.format(dataset_type.value, NETWORK_encoding_type.value, run, str(NETWORK_kernel_variation), number_adversarial_data), 'r') as file:
                    training_data = json.load(file)
                number_samples.append(number_adversarial_data)
                accuracies.append(training_data['accuracy'])
                max_number_for_dataset = number_adversarial_data
            except adversarial_dataset.NotEnoughAdversarialData:
                print('Not enough adversarial data - stopping for {} dataset'.format(dataset_type.value))
                print()
                break
        results['adversarial_data_augmentation_{}'.format(dataset_type.value)] = {'dataset': dataset_type.value,
                                                                                  'number': number_samples,
                                                                                  'accuracy': accuracies,
                                                                                  }

        number_samples = []
        accuracies = []
        for number_random_data in range(0, max_number_for_dataset+INCREASE_DATA_AUGMENTATION, INCREASE_DATA_AUGMENTATION):
            print('Random Data Augmentation ({}): {}'.format(dataset_type.value, number_random_data))
            network = random_data_augmentation_trained_network.get_network(dataset_type, NETWORK_encoding_type, number_random_data, NETWORK_kernel_variation, cache_prefix=run)
            with open('./models/{}-{}-{}trained-with-random-data-augmentation-{}-{}.metadata'.format(dataset_type.value, NETWORK_encoding_type.value, run, str(NETWORK_kernel_variation), number_random_data), 'r') as file:
                training_data = json.load(file)
            number_samples.append(number_random_data)
            accuracies.append(training_data['accuracy'])
        results['random_data_augmentation_{}'.format(dataset_type.value)] = {'dataset': dataset_type.value,
                                                                             'number': number_samples,
                                                                             'accuracy': accuracies,
                                                                             }

    # Save results
    with open('./adevrsarial_data_augmentation/{}.json'.format(run), 'w') as file:
        json.dump(results, file, indent=4,)

# kate: replace-tabs true; indent-width 4; indent-mode python;
