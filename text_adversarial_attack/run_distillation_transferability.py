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
import copy
import json

import trained_networks
import dataset
import encode_sentence

import numpy
import tensorflow
import keras.backend

# Constants
TEMPERATURE = 20

# Basic checks
print()
print('--- Basic checks ---')
if not os.path.exists('./adversarial_example/'):
    os.exit('Please create adversarial examples first!')

# Start session
print()
print('--- Start session ---')
session = tensorflow.InteractiveSession()
keras.backend.set_session(session)


# Load networks
print()
print('--- Load networks ---')
trec_network = trained_networks.get_defensively_distilled_network(dataset.DatasetType.TREC, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0, temperature=TEMPERATURE)
ag_network = trained_networks.get_defensively_distilled_network(dataset.DatasetType.AG, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0, temperature=TEMPERATURE)
amazonmovie_network = trained_networks.get_defensively_distilled_network(dataset.DatasetType.AMAZONMOVIE, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0, temperature=TEMPERATURE)


# Prepare containers and word2vec model
print()
print('--- Preperations ---')
if not os.path.exists('./distillation_transferability-{}/'.format(TEMPERATURE)):
    os.makedirs('./distillation_transferability-{}/'.format(TEMPERATURE))

encode_sentence.load_word2vec()

trec_results = dict()
trec_results['wrong_classification'] = list()  # List of ids for which the networks create a wrong classification on the normal input
trec_results['transferability'] = list()  # Should contain [id, successful, output_class]

ag_results = copy.deepcopy(trec_results)  # Reuse the dict structure
amazonmovie_results = copy.deepcopy(trec_results)

trec_id = list()
trec_input = list()
trec_target_class = list()
trec_adversarial_input = list()

ag_id = list()
ag_input = list()
ag_target_class = list()
ag_adversarial_input = list()

amazonmovie_id = list()
amazonmovie_input = list()
amazonmovie_target_class = list()
amazonmovie_adversarial_input = list()


# Get all adversarial example data
print()
print('--- Get adversarial example data ---')
i = 0
number = len(os.listdir('./adversarial_example/'))
for filename in os.listdir('./adversarial_example/'):
    i = i+1
    print('File {0:5d} of {1:5d}'.format(i, number), end='\r')
    if not filename.endswith('.json'):
        continue
    dataset_type = filename.split('-')[0]
    filename = './adversarial_example/' + filename

    with open(filename, 'r') as file:
        file_content = json.load(file)

    if dataset_type == 'trec':
        if file_content['dataset'] != 'trec':
            print('Dataset in {} does not match file name'.format(filename))
            continue
        trec_id.append(file_content['input_index'])
        trec_input.append(file_content['input_sentence'])
        trec_target_class.append(file_content['input_class'])
        trec_adversarial_input.append(file_content['adversarial_example'])
    elif dataset_type == 'ag':
        if file_content['dataset'] != 'ag':
            print('Dataset in {} does not match file name'.format(filename))
            continue
        ag_id.append(file_content['input_index'])
        ag_input.append(file_content['input_sentence'])
        ag_target_class.append(file_content['input_class'])
        ag_adversarial_input.append(file_content['adversarial_example'])
    elif dataset_type == 'amazonmovie':
        if file_content['dataset'] != 'amazonmovie':
            print('Dataset in {} does not match file name'.format(filename))
            continue
        amazonmovie_id.append(file_content['input_index'])
        amazonmovie_input.append(file_content['input_sentence'])
        amazonmovie_target_class.append(file_content['input_class'])
        amazonmovie_adversarial_input.append(file_content['adversarial_example'])
    else:
        print('Unknown dataset type {} for file {}'.format(dataset_type, filename))

print()


# Get all adversarial example data
print()
print('--- Test input data ---')
print()

print('trec')
trec_used_adversarials = list()
trec_used_adversarials_original_class = list()
trec_used_adversarials_id = list()

for i in range(len(trec_input)):
    print('{0:5d} of {1:5d}'.format(i+1, len(trec_input)), end='\r')
    trec_predicted = trec_network.predict(numpy.asarray([encode_sentence.encode_word2vec(trec_input[i], cache_model=True)]))
    if trec_target_class[i] == int(numpy.argmax(trec_predicted)):
        trec_used_adversarials.append(trec_adversarial_input[i])
        trec_used_adversarials_original_class.append(trec_target_class[i])
        trec_used_adversarials_id.append(trec_id[i])
    else:
        trec_results['wrong_classification'].append(trec_id[i])
print()

print('ag')
ag_used_adversarials = list()
ag_used_adversarials_original_class = list()
ag_used_adversarials_id = list()

for i in range(len(ag_input)):
    print('{0:5d} of {1:5d}'.format(i+1, len(ag_input)), end='\r')
    ag_predicted = ag_network.predict(numpy.asarray([encode_sentence.encode_word2vec(ag_input[i], cache_model=True)]))
    if ag_target_class[i] == int(numpy.argmax(ag_predicted)):
        ag_used_adversarials.append(ag_adversarial_input[i])
        ag_used_adversarials_original_class.append(ag_target_class[i])
        ag_used_adversarials_id.append(ag_id[i])
    else:
        ag_results['wrong_classification'].append(ag_id[i])
print()

print('amazonmovie')
amazonmovie_used_adversarials = list()
amazonmovie_used_adversarials_original_class = list()
amazonmovie_used_adversarials_id = list()

for i in range(len(amazonmovie_input)):
    print('{0:5d} of {1:5d}'.format(i+1, len(amazonmovie_input)), end='\r')
    amazonmovie_predicted = amazonmovie_network.predict(numpy.asarray([encode_sentence.encode_word2vec(amazonmovie_input[i], cache_model=True)]))
    if amazonmovie_target_class[i] == int(numpy.argmax(amazonmovie_predicted)):
        amazonmovie_used_adversarials.append(amazonmovie_adversarial_input[i])
        amazonmovie_used_adversarials_original_class.append(amazonmovie_target_class[i])
        amazonmovie_used_adversarials_id.append(amazonmovie_id[i])
    else:
        amazonmovie_results['wrong_classification'].append(amazonmovie_id[i])
print()


# Test the adversarials
print()
print('--- Test adversarial data ---')

print('trec')
for i in range(len(trec_used_adversarials)):
    print('{0:5d} of {1:5d}'.format(i+1, len(trec_used_adversarials)), end='\r')
    trec_adversarial_predicted = trec_network.predict(numpy.asarray([encode_sentence.encode_word2vec(trec_used_adversarials[i], cache_model=True)]))
    output_class = int(numpy.argmax(trec_adversarial_predicted))
    trec_results['transferability'].append([trec_used_adversarials_id[i], output_class != trec_used_adversarials_original_class[i], output_class])
print()

print('ag')
for i in range(len(ag_used_adversarials)):
    print('{0:5d} of {1:5d}'.format(i+1, len(ag_used_adversarials)), end='\r')
    ag_adversarial_predicted = ag_network.predict(numpy.asarray([encode_sentence.encode_word2vec(ag_used_adversarials[i], cache_model=True)]))
    output_class = int(numpy.argmax(ag_adversarial_predicted))
    ag_results['transferability'].append([ag_used_adversarials_id[i], output_class != ag_used_adversarials_original_class[i], output_class])
print()

print('amazonmovie')
for i in range(len(amazonmovie_used_adversarials)):
    print('{0:5d} of {1:5d}'.format(i+1, len(amazonmovie_used_adversarials)), end='\r')
    amazonmovie_adversarial_predicted = amazonmovie_network.predict(numpy.asarray([encode_sentence.encode_word2vec(amazonmovie_used_adversarials[i], cache_model=True)]))
    output_class = int(numpy.argmax(amazonmovie_adversarial_predicted))
    amazonmovie_results['transferability'].append([amazonmovie_used_adversarials_id[i], output_class != amazonmovie_used_adversarials_original_class[i], output_class])
print()


# Saving results
print()
print('--- Saving results ---')

with open('./distillation_transferability-{}/trec.json'.format(TEMPERATURE), 'w') as file:
    json.dump(trec_results, file, indent=4,)

with open('./distillation_transferability-{}/ag.json'.format(TEMPERATURE), 'w') as file:
    json.dump(ag_results, file, indent=4,)

with open('./distillation_transferability-{}/amazonmovie.json'.format(TEMPERATURE), 'w') as file:
    json.dump(amazonmovie_results, file, indent=4,)


# Finishing
print()
print('--- Finishing ---')

encode_sentence.unload_word2vec()

print()
print('--- Done ---')
print()

# kate: replace-tabs true; indent-width 4; indent-mode python;
