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

for filename in sorted(os.listdir('./adversarial_transferability_retest/')):
    if not filename.endswith('.json'):
        continue

    with open('./adversarial_transferability_retest/{}'.format(filename), 'r') as file:
        results = json.load(file)

    for tag in ['trec_transferability_retrained', 'trec_transferability_second_half', 'trec_transferability_alternative', 'trec_transferability_character', 'ag_transferability_retrained', 'ag_transferability_second_half', 'ag_transferability_alternative', 'ag_transferability_character', 'amazonmovie_transferability_retrained', 'amazonmovie_transferability_second_half', 'amazonmovie_transferability_alternative', 'amazonmovie_transferability_character']:
        if 'transferability' not in tag:
            continue

        number_correct = len(results[tag])
        number_false = len(results['{}_wrong_classification'.format(tag.split('_')[0])])

        transferability_successful = 0
        transferability_unsuccessful = 0
        for data in results[tag]:
            if data[1]:
                transferability_successful += 1
            else:
                transferability_unsuccessful += 1


        print('Results {} (Run {})'.format(tag, filename))
        print('   Number tested: {}'.format(number_correct + number_false))
        print('   Number correctly classified (used later): {}'.format(number_correct))
        print('   Number falsely classified: {}'.format(number_false))
        print('   Transferability rate: {}'.format(transferability_successful / (transferability_successful + transferability_unsuccessful)))
        print('      Number transfered sucessfully: {}'.format(transferability_successful))
        print('      Number transfered unsucessfully: {}'.format(transferability_unsuccessful))

# kate: replace-tabs true; indent-width 4; indent-mode python;

