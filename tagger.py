# The tagger.py starter code for CSC384 A4.

import os
import sys

import numpy as np
from collections import Counter
np.seterr(divide = 'ignore')

UNIVERSAL_TAGS = [
    "VERB",
    "NOUN",
    "PRON",
    "ADJ",
    "ADV",
    "ADP",
    "CONJ",
    "DET",
    "NUM",
    "PRT",
    "X",
    ".",
]

N_tags = len(UNIVERSAL_TAGS)

def read_data_train(path):
    return [tuple(line.split(' : ')) for line in open(path, 'r').read().split('\n')[:-1]]

def read_data_test(path):
    return open(path, 'r').read().split('\n')[:-1]

def read_data_ind(path):
    return [int(line) for line in open(path, 'r').read().split('\n')[:-1]]

def write_results(path, results):
    with open(path, 'w') as f:
        f.write('\n'.join(results))

def train_HMM(train_file_name):
    """
    Estimate HMM parameters from the provided training data.

    Input: Name of the training files. Two files are provided to you:
            - file_name.txt: Each line contains a pair of word and its Part-of-Speech (POS) tag
            - fila_name.ind: The i'th line contains an integer denoting the starting index of the i'th sentence in the text-POS data above

    Output: Three pieces of HMM parameters stored in LOG PROBABILITIES :

            - prior:        - An array of size N_tags
                            - Each entry corresponds to the prior log probability of seeing the i'th tag in UNIVERSAL_TAGS at the beginning of a sequence
                            - i.e. prior[i] = log P(tag_i)

            - transition:   - A 2D-array of size (N_tags, N_tags)
                            - The (i,j)'th entry stores the log probablity of seeing the j'th tag given it is a transition coming from the i'th tag in UNIVERSAL_TAGS
                            - i.e. transition[i, j] = log P(tag_j|tag_i)

            - emission:     - A dictionary type containing tuples of (str, str) as keys
                            - Each key in the dictionary refers to a (TAG, WORD) pair
                            - The TAG must be an element of UNIVERSAL_TAGS, however the WORD can be anything that appears in the training data
                            - The value corresponding to the (TAG, WORD) key pair is the log probability of observing WORD given a TAG
                            - i.e. emission[(tag, word)] = log P(word|tag)
                            - If a particular (TAG, WORD) pair has never appeared in the training data, then the key (TAG, WORD) should not exist.

    Hints: 1. Think about what should be done when you encounter those unseen emission entries during deccoding.
           2. You may find Python's builtin Counter object to be particularly useful
    """

    pos_data = read_data_train(train_file_name+'.txt')
    sent_inds = read_data_ind(train_file_name+'.ind')

    # Building tables to store function return data
    prior = np.zeros(N_tags)
    transition = np.zeros((N_tags, N_tags))
    emission = dict()

    # Dictionary to keep count
    tag_word_count = dict()
    prior_count = _build_empty_dict(UNIVERSAL_TAGS)
    sentences = _build_sentences(sent_inds, pos_data)
    # Fill prior table
    total = 0
    for sentence in sentences:
        for i in range(N_tags):
            if sentence[0][1] == UNIVERSAL_TAGS[i]:
                prior_count[sentence[0][1]] += 1
                total += 1

    for pos_tag in UNIVERSAL_TAGS:
        t_index = UNIVERSAL_TAGS.index(pos_tag)
        if prior_count[pos_tag] == 0:
            prior[t_index] = np.log(0.000000000000000000000001)
        else:
            prior[t_index] = np.log(prior_count[pos_tag]/total)

    # Fill trans table
    for sentence in sentences:
        for s1, s2 in _chain(sentence):
            tag_from_index = UNIVERSAL_TAGS.index(s1[1])
            tag_to_index = UNIVERSAL_TAGS.index(s2[1])
            transition[tag_from_index, tag_to_index] += 1

    transition = _sum_up_transitions(transition)


    # Fill emissions table
    #tags_count stores num of each tag seen, indexed according to universal_tags
    tags_count = np.zeros(N_tags)
    for word, pos_tag in pos_data:
        tag_word_count.setdefault((pos_tag, word), 0)
        tag_word_count[(pos_tag, word)] += 1
        index = UNIVERSAL_TAGS.index(pos_tag)
        tags_count[index] += 1

    for key in tag_word_count.keys():
        pos_tag = key[0]
        word = key[1]
        pair_count = tag_word_count[key]
        index = UNIVERSAL_TAGS.index(pos_tag)
        tag_count = tags_count[index]
        prob = np.log(pair_count/tag_count)
        emission[(pos_tag, word)] = prob

    ####################
    # STUDENT CODE HERE
    ####################

    return prior, transition, emission


def _sum_up_transitions(transition):
    current_index = 0
    for row in transition:
        temp = []
        for num in row:
            prob = np.log(num/sum(row))
            temp.append(prob)
        transition[current_index] = temp
        current_index += 1
    transition = np.array(transition)
    return transition


def _build_sentences(sent_inds, pos_data):
    sentences = []
    for index in range(len(sent_inds)-1):
        sen_start = sent_inds[index]
        sen_end = sent_inds[index+1]
        sentences.append(pos_data[sen_start:sen_end])
    final = sent_inds[-1]
    sentences.append(pos_data[final:])
    return sentences


def _chain(s1):
    out = []
    for i in range(len(s1)-1):
        t1 = s1[i]
        t2 = s1[i+1]
        out.append((t1, t2))
    return out


def _build_empty_dict(keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = 0
    return new_dict


def tag(train_file_name, test_file_name):
    """
    Train your HMM model, run it on the test data, and finally output the tags.
    Your code should write the output tags to (test_file_name).pred, where each line contains a POS tag as in UNIVERSAL_TAGS
    """
    results = []
    prior, transition, emission = train_HMM(train_file_name)
    pos_data = read_data_test(test_file_name+'.txt')
    sent_inds = read_data_ind(test_file_name+'.ind')
    ####################
    # STUDENT CODE HERE
    ####################
    sentences = _build_sentences(sent_inds, pos_data)

    for sentence in sentences:
        prob_trellis, path_trellis = viterbi(sentence, UNIVERSAL_TAGS, prior, transition, emission)
        trellis_path = path_trellis[np.argmax(prob_trellis[:, -1])][-1]
        results += trellis_path
    write_results(test_file_name+'.pred', results)


def viterbi(O, S, initial_probs, transition_matrix, emission_matrix):
    prob_trellis = np.zeros([N_tags, len(O)])
    path_trellis = np.zeros([N_tags, len(O)], dtype=list)
    # Determine trellis values for X1
    for s in range(N_tags):
        prob_trellis[s, 0] = initial_probs[s] + emission_matrix.setdefault((S[s], O[0]), np.log(0.00000001))
        path_trellis[s, 0] = [S[s]]
    # For X2-XT find each current state's most likely prior state x
    for o in range(1, len(O)):
        for s in range(N_tags):
            x = np.argmax(prob_trellis[:, o-1] + transition_matrix[:, s])
            prob_trellis[s, o] = prob_trellis[x, o-1] + transition_matrix[x, s] + emission_matrix.setdefault((S[s], O[o]), np.log(0.00000001))
            path_trellis[s, o] = path_trellis[x, o-1] + [S[s]]
    return prob_trellis, path_trellis


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")
    # Tagger expects the input call: "python3 tagger.py -d <training file> -t <test file>"
    # E.g. python3 tagger.py -d data/train-public -t data/test-public-small
    parameters = sys.argv
    train_file_name = parameters[parameters.index("-d")+1]
    test_file_name = parameters[parameters.index("-t")+1]

    # Start the training and tagging operation.
    tag (train_file_name, test_file_name)
