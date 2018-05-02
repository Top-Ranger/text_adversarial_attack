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
import sys
import statistics
import collections
import subprocess

import dataset

GNUPLOT_FILE = '''set terminal pdf font "Helvetica,24"
set output '{0}-{1}.pdf'

set boxwidth 0.5
set style fill transparent solid 0.75 noborder
set title "Number of changes ({0}, T={1})"
set xlabel "Number of changes"
set ylabel "Number of examples"
set logscale y 10
set yrange [0.9:]
set xtics rotate by -45
set grid noxtics ytics

plot "{0}-{1}.dat" using 1:2 with boxes lc rgb "green" title "Adversarial examples"
'''

temperatures = []

if not os.path.exists('./analysis_distillation_adversarial_example/'):
    os.makedirs('./analysis_distillation_adversarial_example/')

# Find temperatures
for file in os.listdir('./'):
    if file.startswith('distillation_adversarial_example_input_unusable-'):
        temperatures.append(file.split('-')[1])

if len(temperatures) == 0:
    print('No output folder found')
    sys.exit()

temperatures.sort()

for temp in temperatures:
    successful = dict()
    unsuccessful = dict()
    running_time = dict()
    changes_list = dict()

    sentence_length_successful = dict()

    for dataset_type in list(dataset.DatasetType):
        successful[dataset_type.value] = 0
        unsuccessful[dataset_type.value] = 0
        running_time[dataset_type.value] = 0.0
        changes_list[dataset_type.value] = []

        sentence_length_successful[dataset_type.value] = 0

    # Unsuccessful
    for file in os.listdir('./distillation_adversarial_example_unsuccessful-{}'.format(temp)):
        for dataset_type in list(dataset.DatasetType):
            if file.startswith(dataset_type.value):
                unsuccessful[dataset_type.value] += 1

    # Successful
    for file in os.listdir('./distillation_adversarial_example-{}/'.format(temp)):
        for dataset_type in list(dataset.DatasetType):
            if file.startswith(dataset_type.value):
                successful[dataset_type.value] += 1

                # Decode JSON
                result = dict()
                with open('./distillation_adversarial_example-{}/{}'.format(temp, file)) as result_file:
                    result = json.load(result_file)
                changes_list[dataset_type.value].append(result['number_changes'])
                running_time[dataset_type.value] += result['running_time']
                sentence_length_successful[dataset_type.value] += len(result['input_sentence'].split(' '))

    # Show results
    for dataset_type in list(dataset.DatasetType):
        print('Results {} - Temperature {}:'.format(dataset_type.value, temp))
        print('   Number total: {}'.format(successful[dataset_type.value] + unsuccessful[dataset_type.value]))
        print('   Number successful: {}'.format(successful[dataset_type.value]))
        print('   Number unsuccessful: {}'.format(unsuccessful[dataset_type.value]))
        if unsuccessful[dataset_type.value] != 0:
            print('   Success rate: {}'.format(successful[dataset_type.value] / (successful[dataset_type.value] + unsuccessful[dataset_type.value])))
        else:
            print('   Success rate: Perfect')
        print('   Average number of changes: {}'.format(statistics.mean(changes_list[dataset_type.value])))
        print('   Median number of changes: {}'.format(statistics.median(changes_list[dataset_type.value])))
        try:
            print('   Mode number of changes: {}'.format(statistics.mode(changes_list[dataset_type.value])))
        except statistics.StatisticsError:
            counter = collections.Counter(changes_list[dataset_type.value])
            max_count = max(counter.values())
            mode_list = []
            for key, value in counter.items():
                if value == max_count:
                    mode_list.append(key)
            print('   Mode number of changes: {}'.format(mode_list))
        print('   Average running time: {}'.format(running_time[dataset_type.value] / successful[dataset_type.value]))
        print('   Average input length for successful runs: {}'.format(sentence_length_successful[dataset_type.value] / successful[dataset_type.value]))

    # Draw plots
    for dataset_type in list(dataset.DatasetType):
        counter = collections.Counter(changes_list[dataset_type.value])
        with open('./analysis_distillation_adversarial_example/{}-{}.dat'.format(dataset_type.value, temp), 'w') as file:
            for i in range(1, max(changes_list[dataset_type.value]) + 1):
                file.write('{} {}\n'.format(i, counter[i]))
        with open('./analysis_distillation_adversarial_example/{}-{}.gnu'.format(dataset_type.value, temp), 'w') as file:
            file.write(GNUPLOT_FILE.format(dataset_type.value, temp))
        subprocess.Popen(['/usr/bin/gnuplot', '{}-{}.gnu'.format(dataset_type.value, temp)],  cwd='./analysis_distillation_adversarial_example/')

# kate: replace-tabs true; indent-width 4; indent-mode python;
