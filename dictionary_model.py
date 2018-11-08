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


def train_cooccurrence_model(training_data_filename, output_filename, threshold=0):
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
    positive, negative = set(), set()
    for word, polarity in zip(word_list, polarities):
        # print('Word:', a, 'Polarity:', b)
        if polarity < (- threshold):
            negative.add(word)
        if polarity > threshold:
            positive.add(word)

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


def get_sentiment(review):
    '''
    Returns a tuple containing the sentiment of the given review and updated
    counts for the number of positive and negative reviews after processing
    the given review.
    '''
    sentiment = 'n'  # neutral
    num_stars = review["stars"]
    if num_stars >= 4:
        sentiment = '+'
    elif num_stars <= 2:
        sentiment = '-'
    return sentiment


def test_model(trained_output_filename, test_data_filename):
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
        sentiment = get_sentiment(review)
        if sentiment is 'n':
            continue
        model_guess = guess(positive, negative, review)

        if sentiment is '+':
            total_pos += 1
            if model_guess is '+':
                correct_pos += 1
            else:
                false_neg.append(review)  # TODO trim content of review (maybe earlier in process)

        if sentiment is '-':
            total_neg += 1
            if model_guess is '-':
                correct_neg += 1
            else:
                false_pos.append(review)
    if total_pos != 0:
        accuracy_pos = correct_pos / total_pos
    else:
        accuracy_pos = 1  # TODO was getting division by 0 error - is this okay solution?
    if total_neg != 0:
        accuracy_neg = correct_neg / total_neg
    else:
        accuracy_neg = 1  # TODO see above
    accuracy_total = (correct_pos + correct_neg) / (total_pos + total_neg)

    accuracies = (accuracy_total, accuracy_pos, accuracy_neg)
    errors = (false_pos, false_neg)
    return accuracies, errors


# TESTING
training_data_file = 'training_data/yelp_training_sample_1000.json'
output_file = 'trained_dictionary_output/trained_conjunction_model_1000.json'
test_data_file = 'test_data/yelp_test_sample_1000.json'
train_conjunction_model(training_data_file, output_file)
output_file_2 = 'trained_dictionary_output/trained_cooccurrence_model_1000.json'
train_cooccurrence_model(training_data_file, output_file_2)

accuracies, errors = test_model(output_file, test_data_file)
accuracy_total, accuracy_pos, accuracy_neg = accuracies
print('-----CONJUNCTIVE-----')
print('Total Accuracy: ', accuracy_total)
print('Positive Accuracy: ', accuracy_pos)
print('Negative Accuracy: ', accuracy_neg)
print('---------------------')


accuracies, errors = test_model(output_file_2, test_data_file)
accuracy_total, accuracy_pos, accuracy_neg = accuracies
print('-----CO-OCCURRENCE-----')
print('Total Accuracy: ', accuracy_total)
print('Positive Accuracy: ', accuracy_pos)
print('Negative Accuracy: ', accuracy_neg)
print('---------------------')
