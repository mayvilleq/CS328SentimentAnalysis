'''
Contains functions that train the conjunction and co-occurence model and
provides a function for testing the trained models.
'''

import json
import random
import string
from math import log
from nltk import pos_tag, word_tokenize
from seed_dictionaries import (
    negative_seeds, positive_seeds, stop_words,
)

# Valids of part of speech - only look at adverbs, adjectives
valid_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'UH']
nouns = ['NN', 'NNS', 'NNP', 'NNPS']  # Include nouns for conjunctive model


def train_conjunction_model(training_data_filename, output_filename):
    '''
    Trains the conjunction dictionary model based on the data provided in the given training
    data file. Writes the components of the trained model out to the given output file.
    '''
    with open(training_data_filename) as data_file:
        reviews = json.loads(data_file.read())

    positive, negative = {}, {}  # context specific sentiment dictionaries

    # Get and proccess text of each review
    for review in reviews:
        words = word_tokenize(review['text'])
        parts_of_speech = pos_tag(words)
        words = negations(words)

        # Check whether seed word occurs
        for i, word in enumerate(words):
            normalized_word = normalize_word(word)
            seed_sentiment = in_seed_dic(normalized_word)

            # If seed word, get previous and next word
            if seed_sentiment:
                if i != 0:
                    prev_word = normalize_word(words[i-1])
                else:
                    prev_word = None
                if i != len(words) - 1:
                    next_word = normalize_word(words[i+1])
                else:
                    next_word = None

                # If previous word is valid, add to corresponding dictionary
                if prev_word and all((
                    prev_word not in string.punctuation,
                    prev_word not in stop_words,
                    parts_of_speech[i-1][1] in (valid_pos + nouns),
                )):
                    if seed_sentiment == '-':
                        if negative.get(prev_word) is None:
                            negative[prev_word] = 1
                        else:
                            negative[prev_word] += 1
                    else:
                        if positive.get(prev_word) is None:
                            positive[prev_word] = 1
                        else:
                            positive[prev_word] += 1

                # If next word is valid, add to corresponding dictionary
                if next_word and all((
                    next_word not in string.punctuation,
                    next_word not in stop_words,
                    parts_of_speech[i+1][1] in (valid_pos + nouns),
                )):
                    if seed_sentiment == '-':
                        if negative.get(next_word) is None:
                            negative[next_word] = 1
                        else:
                            negative[next_word] += 1
                    else:
                        if positive.get(next_word) is None:
                            positive[next_word] = 1
                        else:
                            positive[next_word] += 1

    # Sort dictionaries by count (ties are broken arbitrarily)
    positive = sorted(positive, key=positive.get)
    negative = sorted(negative, key=negative.get)

    # Trim dictionaries to top 200 (or min dic length) words
    min_length = min(len(positive), len(negative))
    if min_length < 200:
        positive = positive[-min_length:]
        negative = negative[-min_length:]
    else:
        positive = positive[-200:]
        negative = negative[-200:]

    # Write out trained data
    trained_data = {'positive': positive + positive_seeds, 'negative': negative + negative_seeds}
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
    num_pos_reviews, num_neg_reviews = 0, 0  # counts of reviews containing positive and negative seed words
    word_count_pos, word_count_neg = [], []  # counts of reviews in which word (corresponding to word_list) co-occurrs with pos/neg words

    for review in reviews:
        pos_seed, neg_seed = False, False  # does review contain pos/neg seed word?
        original_words = word_tokenize(review['text'])
        words = negations(original_words)

        # Check if a pos/neg seed word in review
        for word in words:
            word = normalize_word(word)
            if word is '':
                continue
            if word in positive_seeds and not pos_seed:
                pos_seed = True
                num_pos_reviews += 1
            if word in negative_seeds and not neg_seed:
                neg_seed = True
                num_neg_reviews += 1
            if pos_seed and neg_seed:
                break

        found = {}  # track whether words found in review to avoid double counting

        # Update co-occurrence word counts if seed word in review
        if pos_seed or neg_seed:
            parts_of_speech = pos_tag(original_words)
            for i, word in enumerate(words):
                word = normalize_word(word)

                # Skip invalid words
                if any((
                    word is '',
                    word in stop_words,
                    word in positive_seeds,
                    word in negative_seeds,
                    parts_of_speech[i][1] not in valid_pos,
                    found.get(word, False),  # avoid double counting words
                )):
                    continue

                # Get index of word in word list
                try:
                    index = word_list.index(word)
                except ValueError:
                    word_list.append(word)
                    word_count_pos.append(0)
                    word_count_neg.append(0)
                    index = len(word_list) - 1

                # Update co-occurrence word counts and mark as found
                if pos_seed:
                    word_count_pos[index] += 1
                if neg_seed:
                    word_count_neg[index] += 1
                found[word] = True

    positive, negative = [], []  # context specific sentiment dictionaries
    polarities = compute_polarities(num_pos_reviews, num_neg_reviews, word_count_pos, word_count_neg)

    # Add words to dictionaries based on polarity
    # Only add words that co-occur above a threshold number of times
    for i, polarity in enumerate(polarities):
        if polarity < 0 and word_count_neg[i] > (num_neg_reviews / 300):
            negative.append((word_list[i], polarity))
        if polarity > 0 and word_count_pos[i] > (num_pos_reviews / 300):
            positive.append((word_list[i], polarity))

    # Sort dictionaries by polarity
    positive.sort(key=lambda x: x[1])
    negative.sort(key=lambda x: x[1])

    # Trim dictionaries to top 200 (or min dic length) words
    min_length = min(len(positive), len(negative))
    if min_length < 200:
        positive = positive[-min_length:]
        negative = negative[:min_length]
    else:
        positive = positive[-200:]
        negative = negative[:200]

    # Unzip to remove polarities from dictionary
    positive, _ = zip(*positive)
    negative, _ = zip(*negative)

    # Write out trained data
    trained_data = {'positive': list(positive) + positive_seeds, 'negative': list(negative) + negative_seeds}
    with open(output_filename, 'w') as outfile:
        json.dump(trained_data, outfile)


def compute_polarities(num_pos_reviews, num_neg_reviews, word_count_pos, word_count_neg):
    '''
    Returns a list of word polarities based on the given counts of positive and
    negative reviews and the lists of cooccurence word counts. Order of polarities
    list corresponds to the order of the word count lists.
    '''
    polarities = []
    for pos_count, neg_count in zip(word_count_pos, word_count_neg):

        # Compute odds
        proportion_pos = pos_count / num_pos_reviews
        proportion_neg = neg_count / num_neg_reviews
        odds_pos = proportion_pos / (1 - proportion_pos)
        odds_neg = proportion_neg / (1 - proportion_neg)

        # Compute polarity
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
    return polarities


def guess(positive, negative, review, threshold = 0):
    '''
    Calculates the most probable category for a review to belong to. Returns
    '+' if positive, '-' if negative. (Uses formula for Polarity from Rice-Zorn paper)
    '''
    num_pos_words, num_neg_words = 0, 0
    words = review["text"].split()
    for word in words:
        word = normalize_word(word)
        if word in positive:
            num_pos_words += 1
        if word in negative:
            num_neg_words += 1

    # If polarity 0, random guess
    if (num_pos_words - num_neg_words + threshold) == 0:
        return random.choice(['-', '+'])
    else:
        polarity = (num_pos_words - num_neg_words + threshold) / (num_pos_words + num_neg_words)
        if polarity < 0:
            return '-'
        else:
            return '+'


def in_seed_dic(word):
    '''
    Returns '+' if word is in positive seed dic, '-' if word is in negative seed
    dictionary, and False if it's in neither.
    '''
    if word in negative_seeds:
        return '-'
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


def get_sentiment(review):
    '''
    Returns the sentiment of the given review.
    '''
    sentiment = 'n'  # neutral
    num_stars = review["stars"]
    if num_stars >= 4:
        sentiment = '+'
    elif num_stars <= 2:
        sentiment = '-'
    return sentiment


def negations(word_tokens):
    '''
    Returns a copy of the given word_tokens where any time 'no' or 'not' appears,
    'not' is prefixed to every word before the next punctuation.
    '''
    negated = False
    result = []
    for word in word_tokens:
        if word in ['no', 'not']:
            negated = True
        elif word in string.punctuation:
            negated = False
        elif negated:
            word = 'not-' + word
        result.append(word)
    return result


def test_model(trained_output_filename, test_data_filename, threshold = 0):
    '''
    Tests the model represented by the given trained ouput file against the given
    file of test data. Returns a tuple of accuracy measures and falsely categorized
    reviews. Accuracy measures is a tuple of the total accuracy rate, the accuracy
    on positive reviews, and the accuracies on negative reviews. Falsely categorized
    reviews is a tuple of two lists: false positives and false negatives.
    '''
    positive, negative = load_trained_output(trained_output_filename)
    with open(test_data_filename) as test_data_file:
        reviews = json.loads(test_data_file.read())

    correct_pos, correct_neg = 0, 0  # count of reviews correctly categorized as positive/negative
    total_pos, total_neg = 0, 0  # count of total number of positive/negative reviews
    false_pos, false_neg = [], []  # list of reviews falsely categorized as positive/negative

    for review in reviews:

        # Guess category
        sentiment = get_sentiment(review)
        if sentiment is 'n':
            continue

        model_guess = guess(positive, negative, review)

        # Update counts of correct/incorrect guesses
        if sentiment is '+':
            total_pos += 1
            if model_guess is '+':
                correct_pos += 1
            else:
                false_neg.append(review)
        if sentiment is '-':
            total_neg += 1
            if model_guess is '-':
                correct_neg += 1
            else:
                false_pos.append(review)

    # Compute accuracies
    if total_pos != 0:
        accuracy_pos = correct_pos / total_pos
    else:
        accuracy_pos = "N/A"
    if total_neg != 0:
        accuracy_neg = correct_neg / total_neg
    else:
        accuracy_neg = "N/A"
    accuracy_total = (correct_pos + correct_neg) / (total_pos + total_neg)

    accuracies = (accuracy_total, accuracy_pos, accuracy_neg)
    errors = (false_pos, false_neg)
    return accuracies, errors
