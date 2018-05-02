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
import statistics
import gzip
import sys

import dataset

import numpy


def max_word_distance(x, y):
    x_max = numpy.amax(x, axis=1)
    y_max = numpy.amax(y, axis=1)
    return 0 if numpy.argmax(x_max) == numpy.argmax(y_max) else 1


def euclidean_distance(x, y):
    sum = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            sum += (x[i][j] - y[i][j])**2
    return sum ** (0.5)


'''Format: [('name', function(x,y)), ...]'''
DISTANCE_METHODS = [('maximum word distance', max_word_distance), ('euclidean distance', euclidean_distance)]
TARGET = 'target_network'
TESTED_VARIANTS = ['w2v_retrained', 'w2v_second_half', 'w2v_alternative_first_half']

print('--- Basic checks ---')
if not os.path.exists('./gradients/'):
    sys.exit('Please create gradients first!')

print('--- Analyse gradients ---')
results = dict()
for metric in DISTANCE_METHODS:
    results[metric[0]] = dict()
    for ds in dataset.DatasetType:
        results[metric[0]][ds.value] = dict()
        for variant in TESTED_VARIANTS:
            results[metric[0]][ds.value][variant] = []

number = len(os.listdir('./gradients/'))
i = 0
for filename in os.listdir('./gradients/'):
    i = i+1
    print('File {0:5d} of {1:5d}'.format(i, number), end='\r')
    if not filename.endswith('.json.gz'):
        continue

    with gzip.open('./gradients/{}'.format(filename), 'rt') as file:
        file_content = json.load(file)

    for metric in DISTANCE_METHODS:
        for variant in TESTED_VARIANTS:
            results[metric[0]][file_content['dataset']][variant].append(metric[1](file_content[TARGET], file_content[variant]))
print()

print('--- Results ---')
for ds in dataset.DatasetType:
    for metric in DISTANCE_METHODS:
        print('Distance metric: {}; Dataset type: {}'.format(metric[0], ds.value))
        print('   Mean distances:')
        for variant in TESTED_VARIANTS:
            print('      {}: {}'.format(variant, statistics.mean(results[metric[0]][ds.value][variant])))

# kate: replace-tabs true; indent-width 4; indent-mode python;
