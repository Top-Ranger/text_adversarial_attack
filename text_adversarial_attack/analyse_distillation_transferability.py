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

temperatures = []

# Find temperatures
for file in os.listdir('./'):
    if file.startswith('distillation_transferability-'):
        temperatures.append(file.split('-')[1])

temperatures.sort()

if len(temperatures) == 0:
    sys.exit('No output folder found')

for temp in temperatures:
    for filename in os.listdir('./distillation_transferability-{}/'.format(temp)):
        if not filename.endswith('.json'):
            continue
        dataset_type = filename.split('.')[0]

        with open('./distillation_transferability-{}/'.format(temp) + filename, 'r') as file:
            results = json.load(file)

        transferability_successful = 0
        transferability_unsuccessful = 0
        for data in results['transferability']:
            if data[1]:
                transferability_successful += 1
            else:
                transferability_unsuccessful += 1


        print('Results {} - Temperature {}:'.format(dataset_type, temp))
        print('   Number tested: {}'.format(len(results['transferability']) + len(results['wrong_classification'])))
        print('   Number correctly classified (used later): {}'.format(len(results['transferability'])))
        print('   Number falsely classified: {}'.format(len(results['wrong_classification'])))
        print('   Transferability rate: {}'.format(transferability_successful / (transferability_successful + transferability_unsuccessful)))
        print('      Number transfered sucessfully: {}'.format(transferability_successful))
        print('      Number transfered unsucessfully: {}'.format(transferability_unsuccessful))

# kate: replace-tabs true; indent-width 4; indent-mode python;
