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
import sys
import json
import gzip

import dataset
import trained_networks
import encode_sentence
import gradient

import numpy
import tensorflow
import keras.backend


# Basic checks
print()
print('--- Basic checks ---')
if not os.path.exists('./adversarial_example/'):
    sys.exit('Please create adversarial examples first!')

# Start session
print()
print('--- Start session ---')
session = tensorflow.InteractiveSession()
keras.backend.set_session(session)


# Preparations
print()
print('--- Preperations ---')
if not os.path.exists('./gradients/'):
    os.makedirs('./gradients/')

print('Loading word2vec method into cache')
encode_sentence.load_word2vec()


# Get networks
print()
print('--- Get networks ---')
# TREC
trec_w2v = trained_networks.get_network(dataset.DatasetType.TREC, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0)
trec_w2v_retrained = trained_networks.get_network(dataset.DatasetType.TREC, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0, cache_prefix='retrained-')
trec_w2v_second_half = trained_networks.get_network(dataset.DatasetType.TREC, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=1)
trec_w2v_alternative = trained_networks.get_network(dataset.DatasetType.TREC, dataset.Encoding.WORD2VEC, kernel_variation=[3, 3, 5, 5], every_xth_trainings_data=2, skip_trainings_data=0)
trec_character = trained_networks.get_network(dataset.DatasetType.TREC, dataset.Encoding.CHARACTER, every_xth_trainings_data=2, skip_trainings_data=0)

# AG
ag_w2v = trained_networks.get_network(dataset.DatasetType.AG, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0)
ag_w2v_retrained = trained_networks.get_network(dataset.DatasetType.AG, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0, cache_prefix='retrained-')
ag_w2v_second_half = trained_networks.get_network(dataset.DatasetType.AG, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=1)
ag_w2v_alternative = trained_networks.get_network(dataset.DatasetType.AG, dataset.Encoding.WORD2VEC, kernel_variation=[3, 3, 5, 5], every_xth_trainings_data=2, skip_trainings_data=0)
ag_character = trained_networks.get_network(dataset.DatasetType.AG, dataset.Encoding.CHARACTER, every_xth_trainings_data=2, skip_trainings_data=0)

# AmazonMovie
amazonmovie_w2v = trained_networks.get_network(dataset.DatasetType.AMAZONMOVIE, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0)
amazonmovie_w2v_retrained = trained_networks.get_network(dataset.DatasetType.AMAZONMOVIE, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0, cache_prefix='retrained-')
amazonmovie_w2v_second_half = trained_networks.get_network(dataset.DatasetType.AMAZONMOVIE, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=1)
amazonmovie_w2v_alternative = trained_networks.get_network(dataset.DatasetType.AMAZONMOVIE, dataset.Encoding.WORD2VEC, kernel_variation=[3, 3, 5, 5], every_xth_trainings_data=2, skip_trainings_data=0)
amazonmovie_character = trained_networks.get_network(dataset.DatasetType.AMAZONMOVIE, dataset.Encoding.CHARACTER, every_xth_trainings_data=2, skip_trainings_data=0)


# Calculate the gradients
print('--- Calculate gradients ---')
number_calculations = 0
number = len(os.listdir('./adversarial_example/'))
i = 0
for filename in os.listdir('./adversarial_example/'):
    i = i+1
    print('File {0:5d} of {1:5d}'.format(i, number), end='\r')
    if not filename.endswith('.json'):
        continue
    if os.path.exists('./gradients/{}.gz'.format(filename)):
        continue

    if number_calculations == 10:
        # Get networks
        tensorflow.reset_default_graph()
        session.close()
        keras.backend.clear_session()
        print()
        print('--- Reloading networks ---')
        session = tensorflow.InteractiveSession()
        keras.backend.set_session(session)

        # TREC
        trec_w2v = trained_networks.get_network(dataset.DatasetType.TREC, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0)
        trec_w2v_retrained = trained_networks.get_network(dataset.DatasetType.TREC, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0, cache_prefix='retrained-')
        trec_w2v_second_half = trained_networks.get_network(dataset.DatasetType.TREC, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=1)
        trec_w2v_alternative = trained_networks.get_network(dataset.DatasetType.TREC, dataset.Encoding.WORD2VEC, kernel_variation=[3, 3, 5, 5], every_xth_trainings_data=2, skip_trainings_data=0)
        trec_character = trained_networks.get_network(dataset.DatasetType.TREC, dataset.Encoding.CHARACTER, every_xth_trainings_data=2, skip_trainings_data=0)

        # AG
        ag_w2v = trained_networks.get_network(dataset.DatasetType.AG, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0)
        ag_w2v_retrained = trained_networks.get_network(dataset.DatasetType.AG, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0, cache_prefix='retrained-')
        ag_w2v_second_half = trained_networks.get_network(dataset.DatasetType.AG, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=1)
        ag_w2v_alternative = trained_networks.get_network(dataset.DatasetType.AG, dataset.Encoding.WORD2VEC, kernel_variation=[3, 3, 5, 5], every_xth_trainings_data=2, skip_trainings_data=0)
        ag_character = trained_networks.get_network(dataset.DatasetType.AG, dataset.Encoding.CHARACTER, every_xth_trainings_data=2, skip_trainings_data=0)

        # AmazonMovie
        amazonmovie_w2v = trained_networks.get_network(dataset.DatasetType.AMAZONMOVIE, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0)
        amazonmovie_w2v_retrained = trained_networks.get_network(dataset.DatasetType.AMAZONMOVIE, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=0, cache_prefix='retrained-')
        amazonmovie_w2v_second_half = trained_networks.get_network(dataset.DatasetType.AMAZONMOVIE, dataset.Encoding.WORD2VEC, every_xth_trainings_data=2, skip_trainings_data=1)
        amazonmovie_w2v_alternative = trained_networks.get_network(dataset.DatasetType.AMAZONMOVIE, dataset.Encoding.WORD2VEC, kernel_variation=[3, 3, 5, 5], every_xth_trainings_data=2, skip_trainings_data=0)
        amazonmovie_character = trained_networks.get_network(dataset.DatasetType.AMAZONMOVIE, dataset.Encoding.CHARACTER, every_xth_trainings_data=2, skip_trainings_data=0)
        number_calculations = 0
    else:
        number_calculations += 1

    input_filename = './adversarial_example/' + filename

    with open(input_filename, 'r') as file:
        file_content = json.load(file)
    gradient_file = dict()
    gradient_file['input_sentence'] = file_content['input_sentence']
    gradient_file['dataset'] = file_content['dataset']
    input_sentence_w2v = numpy.asarray([encode_sentence.encode_word2vec(file_content['input_sentence'], cache_model=True)])
    input_sentence_character = numpy.asarray([encode_sentence.encode_character(file_content['input_sentence'])])
    if file_content['dataset'] == dataset.DatasetType.TREC.value:
        gradient_file['target_network'] = gradient.get_gradient(trec_w2v, input_sentence_w2v, session).tolist()[0]
        gradient_file['w2v_retrained'] = gradient.get_gradient(trec_w2v_retrained, input_sentence_w2v, session).tolist()[0]
        gradient_file['w2v_second_half'] = gradient.get_gradient(trec_w2v_second_half, input_sentence_w2v, session).tolist()[0]
        gradient_file['w2v_alternative_first_half'] = gradient.get_gradient(trec_w2v_alternative, input_sentence_w2v, session).tolist()[0]
        gradient_file['character'] = gradient.get_gradient(trec_character, input_sentence_character, session).tolist()[0]
    elif file_content['dataset'] == dataset.DatasetType.AG.value:
        gradient_file['target_network'] = gradient.get_gradient(ag_w2v, input_sentence_w2v, session).tolist()[0]
        gradient_file['w2v_retrained'] = gradient.get_gradient(ag_w2v_retrained, input_sentence_w2v, session).tolist()[0]
        gradient_file['w2v_second_half'] = gradient.get_gradient(ag_w2v_second_half, input_sentence_w2v, session).tolist()[0]
        gradient_file['w2v_alternative_first_half'] = gradient.get_gradient(ag_w2v_alternative, input_sentence_w2v, session).tolist()[0]
        gradient_file['character'] = gradient.get_gradient(ag_character, input_sentence_character, session).tolist()[0]
    elif file_content['dataset'] == dataset.DatasetType.AMAZONMOVIE.value:
        gradient_file['target_network'] = gradient.get_gradient(amazonmovie_w2v, input_sentence_w2v, session).tolist()[0]
        gradient_file['w2v_retrained'] = gradient.get_gradient(amazonmovie_w2v_retrained, input_sentence_w2v, session).tolist()[0]
        gradient_file['w2v_second_half'] = gradient.get_gradient(amazonmovie_w2v_second_half, input_sentence_w2v, session).tolist()[0]
        gradient_file['w2v_alternative_first_half'] = gradient.get_gradient(amazonmovie_w2v_alternative, input_sentence_w2v, session).tolist()[0]
        gradient_file['character'] = gradient.get_gradient(amazonmovie_character, input_sentence_character, session).tolist()[0]
    else:
        print('Unknown dataset type: {}'.format(file_content['dataset']), file=sys.stderr)
        continue

    with gzip.open('./gradients/{}.gz'.format(filename), 'wt') as file:
        json.dump(gradient_file, file)

print()

# kate: replace-tabs true; indent-width 4; indent-mode python;
