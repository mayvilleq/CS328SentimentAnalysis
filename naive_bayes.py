import json
import string


def train_model(training_data_filename, output_filename):
    '''
    Trains the Naive Bayes model based on the data provided in the given training
    data file. Writes the components of the trained model out to the given output file.
    '''
    with open(training_data_filename) as data_file:
        reviews = json.loads(data_file.read())

    word_list = []  # list of all words in reviews
    num_pos, num_neg = 0, 0  # counts of positive and negative reviews
    word_count_pos, word_count_neg = [], []  # counts of words (corresponding to word_list) in pos/neg reviews

    for review in reviews:
        sentiment, num_pos, num_neg = get_sentiment_and_update_counts(review, num_pos, num_neg)
        words = review["text"].split()

        for word in words:
            word = normalize_word(word)
            if word is '':
                continue

            # Get index of word in word list
            if word not in word_list:
                word_list.append(word)
                word_count_pos.append(0)
                word_count_neg.append(0)
                index = len(word_list) - 1
            else:
                index = word_list.index(word)

            # Update sentiment word counts
            if sentiment is '+':
                word_count_pos[index] += 1
            if sentiment is '-':
                word_count_neg[index] += 1

    # Calculate prior
    prob_pos = num_pos / (num_pos + num_neg)
    prob_neg = num_neg / (num_pos + num_neg)
    prior = (prob_pos, prob_neg)

    # Calculate likelihoods
    pos_likelihood, neg_likelihood = [], []
    for i in range(len(word_list)):
        pos_likelihood.append(1 + word_count_pos[i])
        neg_likelihood.append(1 + word_count_neg[i])

    # Normalize likelihoods
    pos_likelihood = [x/sum(pos_likelihood) for x in pos_likelihood]
    neg_likelihood = [x/sum(neg_likelihood) for x in neg_likelihood]
    likelihood = (pos_likelihood, neg_likelihood)

    # Write out trained data
    trained_data = {'word_list': word_list, 'prior': prior, 'likelihood': likelihood}
    with open(output_filename, 'w') as outfile:
        json.dump(trained_data, outfile)
    # for i, word in enumerate(word_list):
    #     print(word, ": ", pos_likelihood[i])
    # print(neg_likelihood)


def get_sentiment_and_update_counts(review, num_pos, num_neg):
    '''
    Returns a tuple containing the sentiment of the given review and updated
    counts for the number of positive and negative reviews after processing
    the given review.
    '''
    sentiment = 'n'  # neutral
    num_stars = review["stars"]
    if num_stars >= 4:
        sentiment = '+'
        num_pos += 1
    elif num_stars <= 2:
        sentiment = '-'
        num_neg += 1
    return sentiment, num_pos, num_neg


def normalize_word(word):
    '''
    Returns a lowercase version of the given word stripped of all whitespace and punctuation
    '''
    word = word.lower().strip()
    for char in string.punctuation:
        word = word.strip(char)
    return word


# TESTING
training_data_file = 'training_data/yelp_training_sample_2.json'
output_file = 'trained_bayes_output/test_2.json'
train_model(training_data_file, output_file)
