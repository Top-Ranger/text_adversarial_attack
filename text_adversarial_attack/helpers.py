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

import urllib
import gensim
import os
import shutil
import logging
import gzip

def get_w2v():
    """
    Loads (and downloads) the Google pretrained word2vec model

    PLEASE note: the resulting model is quite big (5+ GB RAM) and takes some time to decode.

    :returns: gensim.models.KeyedVectors
    """
    if not os.path.exists('./datasets/'):
        os.makedirs('./datasets/')
    if not os.path.exists('./datasets/word2vec/'):
        os.makedirs('./datasets/word2vec/')
    if not os.path.exists('./datasets/word2vec/GoogleNews-vectors-negative300.bin'):
        logging.debug('Downloading word2vec')

        # Find confirm ID
        website = urllib.request.urlopen('https://drive.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM')
        cookie = website.getheader('Set-Cookie').split(';')[0]
        confirm_id = cookie.split('=')[1]
        logging.debug('Using id {}'.format(confirm_id))
        logging.debug('Cookie: {}'.format(cookie))

        # Download file
        gd_request = urllib.request.Request('https://drive.google.com/uc?export=download&confirm={}&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM'.format(confirm_id))
        gd_request.add_header('Cookie', cookie)
        with urllib.request.urlopen(gd_request) as gd_file, open('./datasets/word2vec/GoogleNews-vectors-negative300.bin.gz', 'wb') as output_file:
            shutil.copyfileobj(gd_file, output_file)

        # Unzip file
        logging.debug('Unzipping word2vec file')
        with gzip.open('./datasets/word2vec/GoogleNews-vectors-negative300.bin.gz', 'rb') as zipped, open('./datasets/word2vec/GoogleNews-vectors-negative300.bin', 'wb') as target:
            shutil.copyfileobj(zipped, target)
        os.remove('./datasets/word2vec/GoogleNews-vectors-negative300.bin.gz')

    return gensim.models.KeyedVectors.load_word2vec_format('./datasets/word2vec/GoogleNews-vectors-negative300.bin', binary=True)


def get_character_index_dict():
    """
    Returns a character index dict.

    The dict will contain the index of all normal characters, e.g. a,A=0, b,B=1,...,z,Z=25, digits, special characters and new line. The second return value returns the number of different indices.

    :returns: (dict, int)
    """
    return ({'a': 0,
            'A': 0,
            'b': 1,
            'B': 1,
            'c': 2,
            'C': 2,
            'd': 3,
            'D': 3,
            'e': 4,
            'E': 4,
            'f': 5,
            'F': 5,
            'g': 6,
            'G': 6,
            'h': 7,
            'H': 7,
            'i': 8,
            'I': 8,
            'j': 9,
            'J': 9,
            'k': 10,
            'K': 10,
            'l': 11,
            'L': 11,
            'm': 12,
            'M': 12,
            'n': 13,
            'N': 13,
            'o': 14,
            'O': 14,
            'p': 15,
            'P': 15,
            'q': 16,
            'Q': 16,
            'r': 17,
            'R': 17,
            's': 18,
            'S': 18,
            't': 19,
            'T': 19,
            'u': 20,
            'U': 20,
            'v': 21,
            'V': 21,
            'w': 22,
            'W': 22,
            'x': 23,
            'X': 23,
            'y': 24,
            'Y': 24,
            'z': 25,
            'Z': 25,
            '1': 26,
            '2': 27,
            '3': 28,
            '4': 29,
            '5': 30,
            '6': 31,
            '7': 32,
            '8': 33,
            '9': 34,
            '0': 35,
            ',': 36,
            ';': 37,
            '.': 38,
            '!': 39,
            '?': 40,
            ':': 41,
            '\'': 42,
            '"': 43,
            '&': 44,
            '(': 45,
            ')': 46,
            }, 47)


def number_character_index_dict():
    """
    Get the number of indices in the character index dict.

    Shortcut for get_character_index_dict()[1]

    :returns: int
    """
    return get_character_index_dict()[1]


if __name__ == '__main__':
    print('Loading word2vec model once')
    get_w2v()

# kate: replace-tabs true; indent-width 4; indent-mode python;
