import json
import string
from seed_dictionaries import (
    negative_seeds, positive_seeds
)

#TODO paper says they address negations - prefixed negation terms to all subsequent
#terms until next punctuation mark. We could implement this to improve accuracy
#if we want/have timeself.
#TODO can also look at LIWC like paper does for more seed dictionaries/can
#test this dictionary like they did.

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

    #TODO  ASK ANNA - we think symmetric, but unsure if window we classify as 0.
    #Decide threshold
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
