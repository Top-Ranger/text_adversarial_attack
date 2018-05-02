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
import subprocess

GNUPLOT_FILE = '''set terminal pdf font "Helvetica,24"
set output '{3}.pdf'

set title "{0}"
set xlabel "{4}"
set ylabel "Accuracy"
set yrange [{1}:{2}]
set xtics rotate by -45
set grid noxtics ytics
set rmargin 4

plot "{3}.dat" using 1:2 with linespoints lc rgb "green" pt 2 ps 0.5 notitle
'''

if not os.path.exists('./analysis_adevrsarial_data_augmentation/'):
    os.makedirs('./analysis_adevrsarial_data_augmentation/')

if not os.path.exists('./adevrsarial_data_augmentation/'):
    sys.exit('Please run the data augmentation experiment first!')

data = dict()

for file in os.listdir('./adevrsarial_data_augmentation/'):
    if not file.endswith('.json'):
        continue
    with open('./adevrsarial_data_augmentation/{}'.format(file)) as result_file:
        result = json.load(result_file)

    for key in result.keys():
        if '_augmentation_' in key:
            if key not in data:
                data[key] = dict()
                data[key]['dataset'] = result[key]['dataset']
            for i in range(len(result[key]['number'])):
                if result[key]['number'][i] not in data[key]:
                    data[key][result[key]['number'][i]] = []
                data[key][result[key]['number'][i]].append(result[key]['accuracy'][i])

for key in data.keys():
    accuracies = []
    with open('./analysis_adevrsarial_data_augmentation/{}.dat'.format(key), 'w') as data_file:
        print(key)
        for number_key in sorted([x for x in data[key] if x != 'dataset']):
            print('   {0:5d}: {1}'.format(number_key, statistics.mean(data[key][number_key])))
            data_file.write('{} {}\n'.format(number_key, statistics.mean(data[key][number_key])))
            accuracies.append(statistics.mean(data[key][number_key]))
    print('   Min: {}'.format(min(accuracies)))
    print('   Max: {}'.format(max(accuracies)))
    print('      range: {}'.format(max(accuracies)-min(accuracies)))
    with open('./analysis_adevrsarial_data_augmentation/{}.gnu'.format(key), 'w') as gnuplot_file:
        gnuplot_file.write(GNUPLOT_FILE.format(data[key]['dataset'], min(accuracies)-0.1, max(accuracies)+0.1, key, "Number randomly augmented samples" if "random" in key else "Number adversarial examples"))
    subprocess.Popen(['/usr/bin/gnuplot', '{}.gnu'.format(key)],  cwd='./analysis_adevrsarial_data_augmentation/')

# kate: replace-tabs true; indent-width 4; indent-mode python;
