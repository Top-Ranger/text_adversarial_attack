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
import urllib.request
import json
import logging


def _build_dataset():
    if os.path.exists('./cache/typos/'):  # Dataset is already build
        return

    if not os.path.exists('./datasets/'):
        os.makedirs('./datasets/')
    if not os.path.exists('./datasets/typo/'):
        os.makedirs('./datasets/typo/')

    # Download from http://www.dcs.bbk.ac.uk/~ROGER/corpora.html
    dataset_list = ['missp.dat', 'holbrook-missp.dat', 'aspell.dat', 'wikipedia.dat']

    for dataset in dataset_list:
        if not os.path.exists('./datasets/typo/{}'.format(dataset)):
            logging.debug('Downloading typo file: {}'.format(dataset))
            urllib.request.urlretrieve('http://www.dcs.bbk.ac.uk/~ROGER/{}'.format(dataset), './datasets/typo/{}'.format(dataset))

    if not os.path.exists('./cache/typos/'):
        os.makedirs('./cache/typos/')

    current_word = ''
    current_typos = set()
    for dataset in dataset_list:
        with open('./datasets/typo/{}'.format(dataset), 'r') as file:
            for line in file:
                line = line.replace('\n', '')  # Remove new lines
                if line[0] == '$':
                    # Save old version
                    if current_word != '' and current_word != '?':  # ?: Filter unknown data
                        with open('./cache/typos/{}'.format(current_word), 'w') as save_file:
                            json.dump(list(current_typos), save_file)
                    current_word = line
                    current_word = current_word[1:]  # Remove '$'
                    # try loading existing data
                    if os.path.exists('./cache/typos/{}'.format(current_word)):
                        with open('./cache/typos/{}'.format(current_word), 'r') as load_file:
                            current_typos = set(json.load(load_file))
                    else:
                        current_typos = set()
                else:
                    current_typos.add(line.split(' ')[0])

    # final save
    if current_word != '':
        with open('./cache/typos/{}'.format(current_word), 'w') as save_file:
            json.dump(list(current_typos), save_file)

def get_typos(word):
    _build_dataset()
    if len(word) == 0 or word[0] == '.':  # Safe guard
        return []
    if os.path.exists('./cache/typos/{}'.format(word)):
        with open('./cache/typos/{}'.format(word), 'r') as load_file:
            return json.load(load_file)
    else:
        return []


if __name__ == '__main__':
    print('Building typo dataset')
    _build_dataset()
    print('Testing')
    print(get_typos('volcanoes'))

# kate: replace-tabs true; indent-width 4; indent-mode python;
