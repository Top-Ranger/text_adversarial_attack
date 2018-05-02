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

import gradient
import encode_sentence
import numpy
import nltk
import nltk.corpus
import random
import os
import copy
import numpy
import keras.models
import tensorflow
import typos


_ATTACKS_AVERAGED_PERCEPTRON_TAGGER_CHECKED = False
_ATTACKS_WORDNET_CHECKED = False


class NoAdversarialExampleFound(Exception):
    pass


def attack_w2v(network: keras.models.Model, input_data: str, session: tensorflow.InteractiveSession, data_x: list, data_y: list, attack_key=None, use_keywords =True, typo_chance=1.0) -> str:
    """
    Runs an adversarial attack against a network which uses the Google word2vec encoding.

    The attack is modelled after the attack described by Suranjana Samanta and Sameep Mehta in "Towards Crafting Text Adversarial Samples".
    https://arxiv.org/abs/1707.02812

    :raises: NoAdversarialExampleFound

    :param network: Network to attack
    :type network: keras.model.Model
    :param input_data: Input sentence
    :type input_data: str
    :param session: Session to run the network under
    :type session: tensorflow.InteractiveSession
    :param data_x: Dataset in normal string form
    :type data_x: [str, str, ...]
    :param data_y: Dataset classes as normal integers
    :type data_y: [int, int, ...]
    :param attack_key: Key to store keyword list for this game. None for no caching.
    :type attack_key: str
    :param use_keywords: If set to True, 'Genre specific keywords' will be used (see Samanta and Mehta)
    :type use_keywords: bool
    :param typo_chance: Chance that typos are included into the search
    :type typo_chance: float
    :returns: String containing the resulting attack, int number of changes, array containing changes
    """
    input_sentence = input_data

    if not os.path.exists('./datasets/nltk/'):
        os.makedirs('./datasets/nltk/')

    keyword_list = _create_keyword_list(data_x, data_y, attack_key)

    number_changes = 0
    changes_list = []


    nltk.data.path = ['./datasets/nltk/']
    global _ATTACKS_WORDNET_CHECKED
    global _ATTACKS_AVERAGED_PERCEPTRON_TAGGER_CHECKED
    if not _ATTACKS_WORDNET_CHECKED:
        nltk.download('wordnet', download_dir='./datasets/nltk/')
        _ATTACKS_WORDNET_CHECKED = True
    if not _ATTACKS_AVERAGED_PERCEPTRON_TAGGER_CHECKED:
        nltk.download('averaged_perceptron_tagger', download_dir='./datasets/nltk/')
        _ATTACKS_AVERAGED_PERCEPTRON_TAGGER_CHECKED = True

    w2v_input = numpy.asarray([encode_sentence.encode_word2vec(input_sentence, cache_model=True)])

    # Calculate origin class and current class
    origin_class = int(numpy.argmax(network.predict(w2v_input)))
    current_class = origin_class

    # Select target class for the keyword list
    target_class = random.randint(0, numpy.max(data_y))
    while target_class == origin_class:
        target_class = random.randint(0, numpy.max(data_y))

    # Calculate priorities based on gradient
    input_gradient = gradient.get_gradient(network, w2v_input, session)

    word_score = numpy.amax(input_gradient, axis=2)[0]

    word_priority_list = list()
    for i in range(input_gradient.shape[1]):
        maximum_value = numpy.argmax(word_score)
        word_priority_list.append(maximum_value)
        word_score[maximum_value] = -99999

    print(input_sentence)

    # IMPORTANT: These two have to be kept in sync!
    # This is to reduce miss-tags later after modifying
    input_words = input_sentence.split(" ")
    for i in range(len(input_words)):
        if input_words[i] == '':  # Avoid empty words for tagging
            input_words[i] = ' '
    input_words_tagged = nltk.pos_tag(input_words)
    for i in range(len(input_words)):
        if input_words[i] == ' ':  # Reconstruct original sentence
            input_words[i] = ''
            input_words_tagged[i] = ('', input_words_tagged[i][1])

    # Helper function for getting scores
    def get_score_of_word(sentence_words, index, batch_size=256):
        """
        Calculates the score (gradient) of given word for all given sentences

        :param sentence_words: Sentences to test
        :type sentence_words: list of lists of words
        :param index: Index of the word to score
        :type index: int
        :param batch_size: Batch size. The larger the batch, the faster the calculation but also the higher the RAM consumption
        :type batch_size: int
        :returns: list containing scores for every senetnce
        """
        length = 0
        for sentence in sentence_words:
            length = max(length, len(sentence))
        result = []

        # Build batches
        number_batches = (len(sentence_words) // batch_size) + 1
        for current_batch in range(number_batches):
            print('Calculating score: {0:3d} of {1:3d}'.format(current_batch+1, number_batches), end='\r')
            sentence_w2v = list()
            for i in range(batch_size*current_batch, batch_size*(current_batch+1)):
                if i < len(sentence_words):
                    sentence_w2v.append(encode_sentence.encode_word2vec(' '.join(sentence_words[i]), min_length=length, cache_model=True))
                else:
                    break
            if len(sentence_w2v) == 0:
                break
            gradient_input = numpy.asarray(sentence_w2v)
            sentence_gradient = gradient.get_gradient(network, gradient_input, session)
            for output_gradient in sentence_gradient:
                result.append(numpy.amax(output_gradient, axis=1)[0])
        print()
        return result

    # Basic sanity tests
    for i in range(len(word_priority_list)):
            assert word_priority_list[i] < len(input_words)

    while origin_class == current_class:  # Always change most important word per round
        print('Remaining words to test: {}'.format(len(word_priority_list)))
        if len(word_priority_list) == 0:  # We have tried changing all words - aborting
            raise NoAdversarialExampleFound()

        current_word = word_priority_list.pop(0)
        current_word_tagged = input_words_tagged[current_word]

        if 'RB' in current_word_tagged[1] and current_word != 0:  # The word is an adverb - workaround to not delete the first word as it might lead to incorrect sentences
            number_changes += 1
            changes_list.append(['deletion', int(current_word), input_words[current_word]])
            input_words.pop(current_word)
            input_words_tagged.pop(current_word)
            for i in range(len(word_priority_list)):  # Update position of all words after the deleted one
                if word_priority_list[i] > current_word:
                    word_priority_list[i] = word_priority_list[i] - 1
        else:
            insert = False
            best_word = input_words[current_word]
            best_prediction = -100.0

            # Insertion
            if 'JJ' in current_word_tagged[1]:
                insert_candidate_set = set()
                for keyword in keyword_list[target_class]:
                    if 'RB' in keyword[1]:
                        insert_candidate_set.add(keyword[0])

                # Find best word
                insert_candidate_set  = list(insert_candidate_set)
                test_data = []
                for test_word in insert_candidate_set:
                    test_input_words = copy.deepcopy(input_words)
                    for single_word in reversed(test_word.split('_')):  # Multiple words are connected by '_', e.g. 'a_lot'
                        test_input_words.insert(current_word, single_word)
                    test_data.append(test_input_words)

                # Don't do anything if no candidate exists
                if len(test_data) != 0:
                    test_score = get_score_of_word(test_data, current_word)
                    for i in range(len(test_data)):
                        if test_score[i] > best_prediction:
                            best_prediction = test_score[i]
                            best_word = insert_candidate_set[i]
                            insert = True

            # Replacement
            candidate_set = set()
            # Keywords for candidates
            if use_keywords:
                for keyword in keyword_list[target_class]:  # Doens't work well at least for TREC - produces unrecognisable questions
                    if keyword[1] == current_word_tagged[1]:
                        candidate_set.add(keyword[0])

            # Synonyms for candidates
            if 'NN' in current_word_tagged[1]:  # Noun
                for synonym_class in nltk.corpus.wordnet.synsets(input_words[current_word], pos=nltk.corpus.wordnet.NOUN):
                    for synonym in synonym_class.lemma_names():
                        candidate_set.add(str(synonym))


            if 'VB' in current_word_tagged[1]:  # Verb
                for synonym_class in nltk.corpus.wordnet.synsets(input_words[current_word], pos=nltk.corpus.wordnet.VERB):
                    for synonym in synonym_class.lemma_names():
                        candidate_set.add(str(synonym))

            if 'JJ' in current_word_tagged[1]:  # Adjective
                for synonym_class in nltk.corpus.wordnet.synsets(input_words[current_word], pos=nltk.corpus.wordnet.ADJ):
                    for synonym in synonym_class.lemma_names():
                        candidate_set.add(str(synonym))

            if 'RB' in current_word_tagged[1]:  # Adverb
                for synonym_class in nltk.corpus.wordnet.synsets(input_words[current_word], pos=nltk.corpus.wordnet.ADV):
                    for synonym in synonym_class.lemma_names():
                        candidate_set.add(str(synonym))

            # Typos for candidates
            if random.random() < typo_chance:
                for typo in typos.get_typos(input_words[current_word]):
                    candidate_set.add(typo)

            # Find best word
            test_data = []
            candidate_set = list(candidate_set)
            for test_word in candidate_set:
                test_input_words = copy.deepcopy(input_words)
                test_input_words.pop(current_word)  # Remove word temporarily - the new one will be inserted here
                for single_word in reversed(test_word.split('_')):  # Multiple words are connected by '_', e.g. 'a_lot'
                    test_input_words.insert(current_word, single_word)
                test_data.append(test_input_words)

            # Don't do anything if no candidate exists
            if len(test_data) != 0:
                test_score = get_score_of_word(test_data, current_word)
                for i in range(len(test_data)):
                    if test_score[i] > best_prediction:
                        best_prediction = test_score[i]
                        best_word = candidate_set[i]
                        insert = False

            if insert:
                number_changes += 1
                changes_list.append(['insertion', int(current_word), best_word])
                for single_word in reversed(best_word.split('_')):  # Multiple words are connected by '_', e.g. 'a_lot'
                    input_words.insert(current_word, single_word)
                    input_words_tagged.insert(current_word, (single_word, 'RB'))
                number_words = 1 + best_word.count('_')
                for i in range(len(word_priority_list)):  # Update position of all words after the inserted one
                    if word_priority_list[i] > current_word:
                        word_priority_list[i] = word_priority_list[i] + number_words
            else:
                if best_word != input_words[current_word]:
                    number_changes += 1
                    changes_list.append(['modification', int(current_word), input_words[current_word], best_word])
                input_words.pop(current_word)  # Remove word temporarily - the new one will be inserted here
                old_pos_class = input_words_tagged.pop(current_word)[1]
                for single_word in reversed(best_word.split('_')):  # Multiple words are connected by '_', e.g. 'a_lot'
                    input_words.insert(current_word, single_word)
                    input_words_tagged.insert(current_word, (single_word, old_pos_class))
                additional_inserted = best_word.count('_')
                for i in range(len(word_priority_list)):  # Update index if more than one word
                    if word_priority_list[i] > current_word:
                        word_priority_list[i] = word_priority_list[i] + additional_inserted

        # Basic sanity tests
        assert len(input_words) == len(input_words_tagged)
        for i in range(len(input_words)):
            assert input_words[i] == input_words_tagged[i][0]
        for i in range(len(word_priority_list)):
            assert word_priority_list[i] < len(input_words)

        # Test modified input
        input_sentence = ' '.join(input_words)
        print(input_sentence)
        w2v_input = numpy.asarray([encode_sentence.encode_word2vec(input_sentence, cache_model=True)])
        current_class = int(numpy.argmax(network.predict(w2v_input)))

    # Sanity check
    assert number_changes == len(changes_list)
    return input_sentence, number_changes, changes_list


def _create_keyword_list(data_x: list, data_y: list, attack_key=None):
    """
    Creates (or loads) the keyword list

    The keyword list is an array with sets for every class
    :param data_x: Dataset in normal string form
    :type data_x: [str, str, ...]
    :param data_y: Dataset classes as normal integers
    :type data_y: [int, int, ...]
    :param attack_key: Key to store keyword list for this game. None for no caching.
    :type attack_key: str
    :returns: Keyword list
    """
    if not os.path.exists('./cache/keyword_list/'):
        os.makedirs('./cache/keyword_list/')

    if attack_key is not None and os.path.exists('./cache/keyword_list/{}/'.format(attack_key)):
        return numpy.load('./cache/keyword_list/{}/keyword_list.npy'.format(attack_key))

    num_class = numpy.max(data_y) + 1

    keyword_list = list()
    for _ in range(num_class):
        keyword_list.append(set())

    nltk.data.path = ['./datasets/nltk/']
    global _ATTACKS_AVERAGED_PERCEPTRON_TAGGER_CHECKED
    if not _ATTACKS_AVERAGED_PERCEPTRON_TAGGER_CHECKED:
        nltk.download('averaged_perceptron_tagger', download_dir='./datasets/nltk/')
        _ATTACKS_AVERAGED_PERCEPTRON_TAGGER_CHECKED = True

    # add all words into keyword sets

    for i in range(len(data_x)):
        current_word_list = data_x[i].split()
        current_word_list = nltk.pos_tag(current_word_list)
        for word in current_word_list:
            keyword_list[data_y[i]].add(word)

    # only use unique words
    temp_keyword_list = copy.deepcopy(keyword_list)
    for current_set in range(len(keyword_list)):
        for difference_set in range(len(keyword_list)):
            if current_set is not difference_set:
                keyword_list[current_set].difference_update(temp_keyword_list[difference_set])

    if attack_key is not None:
        os.makedirs('./cache/keyword_list/{}/'.format(attack_key))
        numpy.save('./cache/keyword_list/{}/keyword_list.npy'.format(attack_key), keyword_list)

    return keyword_list

# kate: replace-tabs true; indent-width 4; indent-mode python;
