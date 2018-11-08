import json
import string
from math import log
from seed_dictionaries import (
    negative_seeds, positive_seeds
)

# TODO paper says they address negations - prefixed negation terms to all subsequent
# terms until next punctuation mark. We could implement this to improve accuracy
# if we want/have timeself.
# TODO can also look at LIWC like paper does for more seed dictionaries/can
# test this dictionary like they did.


def train_conjunction_model(training_data_filename, output_filename):
    '''
    Trains the conjunction dictionary model based on the data provided in the given training
    data file. Writes the components of the trained model out to the given output file.
    '''
    with open(training_data_filename) as data_file:
        reviews = json.loads(data_file.read())

    # TODO do we include seed words in these dictionaries??
    positive, negative = set(), set()  # context specific sentiment dictionaries

    for review in reviews:
        words = review["text"].split()
        for i, word in enumerate(words):
            normalized_word = normalize_word(word)
            if normalized_word is '':
                continue

            seed_sentiment = in_seed_dic(normalized_word)
            # TODO deal with punctuation
            if seed_sentiment:
                if i != 0:
                    prev_word = words[i-1]
                else:
                    prev_word = None
                if i != len(words) - 1:
                    next_word = words[i+1]
                else:
                    next_word = None
                if seed_sentiment == '-':
                    if prev_word and (prev_word.lower() == normalize_word(prev_word)):
                        negative.add(normalize_word(prev_word))
                    if next_word and (word.lower() == normalized_word):
                        negative.add(normalize_word(next_word))
                if seed_sentiment == '+':
                    if prev_word and (prev_word.lower() == normalize_word(prev_word)):
                        positive.add(normalize_word(prev_word))
                    if next_word and (word.lower() == normalized_word):
                        positive.add(normalize_word(next_word))

    # Write out trained data
    trained_data = {'positive': list(positive), 'negative': list(negative)}
    with open(output_filename, 'w') as outfile:
        json.dump(trained_data, outfile)


def train_cooccurrence_model(training_data_filename, output_filename):
    '''
    Trains the co-occurrence dictionary model based on the data provided in the given training
    data file. Writes the components of the trained model out to the given output file.
    '''
    with open(training_data_filename) as data_file:
        reviews = json.loads(data_file.read())

    word_list = []  # list of all words in reviews
    num_pos, num_neg = 0, 0  # counts of reviews containing positive and negative seed words
    word_count_pos, word_count_neg = [], []  # counts of reviews in which word (corresponding to word_list) co-occurrs with pos/neg words

    for review in reviews:
        pos, neg = False, False  # does review contain pos/neg seed word?
        words = review["text"].split()
        # Check if a pos/neg seed word in review
        for word in words:
            word = normalize_word(word)
            if word is '':
                continue
            if word in positive_seeds and not pos:
                pos = True
                num_pos += 1
            if word in negative_seeds and not neg:
                neg = True
                num_neg += 1
            if pos and neg:
                break

        found = {}  # track whether words found  in review to avoid double counting

        # Update word counts if seed word in review
        if pos or neg:
            for word in words:
                word = normalize_word(word)
                if word is '':
                    continue
                if word in positive_seeds or word in negative_seeds:  # don't include seed words
                    continue
                if found.get(word, False):  # avoid double counting words
                    continue

                # Get index of word in word list
                if word not in word_list:
                    word_list.append(word)
                    word_count_pos.append(0)
                    word_count_neg.append(0)
                    index = len(word_list) - 1
                else:
                    index = word_list.index(word)

                # Update co-occurrence word counts
                if pos:
                    word_count_pos[index] += 1
                if neg:
                    word_count_neg[index] += 1
                found[word] = True

    # Compute word polarities
    polarities = []
    for pos_count, neg_count in zip(word_count_pos, word_count_neg):
        proportion_pos = pos_count / num_pos
        proportion_neg = neg_count / num_neg
        odds_pos = proportion_pos / (1 - proportion_pos)
        odds_neg = proportion_neg / (1 - proportion_neg)
        if odds_neg == odds_pos:
            polarity = 0
        elif odds_neg == 0:
            polarity = float('inf')
        elif odds_pos == 0:
            polarity = float('-inf')
        else:
            ratio = log(odds_pos / odds_neg)
            polarity = (pos_count + neg_count) * ratio
        polarities.append(polarity)

    # TODO replace with writing out to file dictionaries (need polarity thresholds)
    for a, b in zip(word_list, polarities):
        print('Word:', a, 'Polarity:', b)


def guess(positive, negative, review, threshold=None):
    '''
    Calculates the most probable category for a review to belong to. Returns
    '+' if positive, '-' if negative. (Uses formula for Polarity from Rice-Zorn paper)
    '''
    if threshold is not None:
        #TODO sort positive and negative list based on threshold if not
        #done in co-occurrence model - left for now - in test.py i assume this
        #is not done here in test.py
        positive = positive
        negative = negative

    num_pos_words, num_neg_words = 0,0
    words = review["text"].split()
    for word in words:
        word = normalize_word(word)
        if word in positive:
            num_pos_words += 1
        if word in negative:
            num_neg_words += 1

    polarity = (num_pos_words - num_neg_words)/(num_pos_words + num_neg_words)

    #TODO is this what threshold is for, or is it for co-occurrence case? Or maybe
    #they're the same?? Will use 0 for the time being.
    if polarity <= 0:
        return '-'
    return '+'


def in_seed_dic(word):
    '''
    Returns '+' if word is in positive seed dic, '-' if word is in negative seed
    dictionary, and False if it's in neither or both.
    '''
    if word in negative_seeds:
        if word not in positive_seeds:
            return '-'
    else:
        if word in positive_seeds:
            return '+'
    return False


def normalize_word(word):
    '''
    Returns a lowercase version of the given word stripped of all whitespace and punctuation.
    '''
    word = word.lower().strip()
    for char in string.punctuation:
        word = word.strip(char)
    return word


def load_trained_output(trained_output_filename):
    '''
    Loads the trained output from the given file name and returns a tuple of the
    corresponding word list, prior, and likelihood.
    '''
    with open(trained_output_filename) as data_file:
        trained_data = json.loads(data_file.read())
        positive = trained_data['positive']
        negative = trained_data['negative']
        return positive, negative


# TESTING
training_data_file = 'training_data/yelp_training_sample_50.json'
output_file = 'trained_dictionary_output/trained_conjunction_model_50.json'
test_data_file = 'test_data/yelp_test_sample_50.json'
train_conjunction_model(training_data_file, output_file)
output_file_2 = 'trained_dictionary_output/trained_cooccurrence_model_50.json'
train_cooccurrence_model(training_data_file, output_file_2)
