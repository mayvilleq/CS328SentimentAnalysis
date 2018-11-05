import json
import string
from seed_dictionaries import (
    negative_seeds, positive_seeds
)

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
                        negative.add(normalize_word(prev_word)))
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


# TESTING
training_data_file = 'training_data/yelp_training_sample_50.json'
output_file = 'trained_dictionary_output/trained_conjunction_model_50.json'
test_data_file = 'test_data/yelp_test_sample_50.json'
train_conjunction_model(training_data_file, output_file)
