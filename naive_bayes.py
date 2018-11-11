import json
import string
from nltk import word_tokenize

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
        words = word_tokenize(review['text'])
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
        word_list = trained_data['word_list']
        prior = trained_data['prior']
        likelihood = trained_data['likelihood']
        return word_list, prior, likelihood


# TODO update with info/stats we want to gather from testing
def test_model(trained_output_filename, test_data_filename):
    '''
    Tests the model represented by the given trained ouput file against the given
    file of test data. Returns a tuple of accuracy measures and falsely categorized
    reviews. Accuracy measures is a tuple of the total accuracy rate, the accuracy
    on positive reviews, and the accuracies on negative reviews. Falsely categorized
    reviews is a tuple of two lists: false positives and false negatives.
    '''
    word_list, prior, likelihood = load_trained_output(trained_output_filename)
    with open(test_data_filename) as test_data_file:
        reviews = json.loads(test_data_file.read())

    correct_pos, correct_neg = 0, 0  # count of reviews correctly categorized as positive/negative
    total_pos, total_neg = 0, 0  # count of total number of positive/negative reviews
    false_pos, false_neg = [], []  # list of reviews falsely categorized as positive/negative

    for review in reviews:
        sentiment, _, _ = get_sentiment_and_update_counts(review, 0, 0)
        if sentiment is 'n':
            continue
        model_guess = guess_2_function(review, word_list, prior, likelihood)

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
        accuracy_pos = "N/A"  # TODO was getting division by 0 error - is this okay solution?
    if total_neg != 0:
        accuracy_neg = correct_neg / total_neg
    else:
        accuracy_neg = "N/A"  # TODO see above
    accuracy_total = (correct_pos + correct_neg) / (total_pos + total_neg)

    accuracies = (accuracy_total, accuracy_pos, accuracy_neg)
    errors = (false_pos, false_neg)
    return accuracies, errors


def guess_function(review, word_list, prior, likelihood):
    '''
    Calculates the most probable category for a review to belong to. Returns
    '+' if positive, '-' if negative. (Uses formula 1 from science direct article)
    '''
    prob_positive = prior[0]
    prob_negative = prior[1]
    likelihood_positive = likelihood[0]
    likelihood_negative = likelihood[1]

    word_vector = review_to_word_vector(review, word_list)

    # Calculate product of likelihoods
    for i, word in enumerate(word_list):
        if word_vector[i] != 0:
            prob_positive *= (likelihood_positive[i])**(word_vector[i])
            prob_negative *= (likelihood_negative[i])**(word_vector[i])

    # Return argmax of two categories
    if prob_positive > prob_negative:
        return '+'
    else:
        return '-'


def guess_2_function(review, word_list, prior, likelihood):
    '''
    guess_2_function() performs same function as guess_function(), but here were try with different
    algorithm (looping through words in review rather than words in word_list), to see if we can
    cut down on runtime
    '''
    prob_positive = prior[0]
    prob_negative = prior[1]
    likelihood_positive = likelihood[0]
    likelihood_negative = likelihood[1]

    words = review["text"].split()

    for word in words:
        word = normalize_word(word)
        if word is '':
            continue
        if word in word_list:
            word_index = word_list.index(word)
            prob_positive *= (likelihood_positive[word_index])
            prob_negative *= (likelihood_negative[word_index])

    # Return argmax of two categories
    if prob_positive > prob_negative:
        return '+'
    else:
        return '-'


def review_to_word_vector(review, word_list):
    '''
    Takes a review (in json dictionary form) in and returns a vector storing
    word counts of the words in word_list in.
    '''

    word_vector = [0 for i in range(len(word_list))]
    words = review["text"].split()
    for word in words:
        word = normalize_word(word)
        if word is not '':
            if word in word_list:
                word_index = word_list.index(word)
                word_vector[word_index] += 1
    return word_vector


def main():
    # TESTING
    training_data_file = 'training_data/yelp_training_sample_1000.json'
    output_file = 'trained_bayes_output/trained_model_1000.json'
    test_data_file = 'test_data/yelp_test_sample_1000.json'
    train_model(training_data_file, output_file)
    accuracies, errors = test_model(output_file, test_data_file)

    '''
    Times for guess_1 vs guess_2 function
    I did not run on training or testing data, instead just used trained_bayes_output files in function test_model
    Timed on my laptop with time.clock(). Added results to make sure functions doing same thing.
    output_file, test_data_file       guess_1 time:   guess_1_results:      guess_2 time:      guess_2_results:
    2,2                                 .046875       (1.0, 1.0, 1)          .09375            (1.0, 1.0, 1)
    2,50                                 .109         (.85106, 1.0, 0.0)     .078125           (.86106.., 1.0, 0.0)
    50,2                                .046875       (1.0, 1.0, 1)          .0625             (1.0, 1.0, 1)
    50,50                               .15625        (.7234, .825, .142)    .21875            (.7234, .825, .142..)
    1500,2                              .09375        (1.0, 1.0, 1)          .09375            (1.0, 1.0, 1)
    1500,50                             .375           (.74, .75, .71)       .265              (.74, .75, .7142)
    500, 500                             2.8125       (.7042, .7329, .613)    2.453125         (.704, .7329, .613)
    1500, 500                            3.6875                               3.84375
    1000, 10000                          6.4375       (.73, .71, .79)        5.078125          (.732, .714, .79)
    1500, 1000                          7.06           (.726, .71, .76)      5.453125          (.726, .71, .76)
    '''

    accuracy_total, accuracy_pos, accuracy_neg = accuracies
    false_pos, false_neg = errors

    print('Total Accuracy: ', accuracy_total)
    print('Positive Accuracy: ', accuracy_pos)
    print('Negative Accuracy: ', accuracy_neg)
    print('---------------------')

    # print(len(false_pos), 'False Positives: ')
    # stars = [x['stars'] for x in false_pos]
    # text = [x['text'] for x in false_pos]
    # for a, b in zip(stars, text):
    #     print('Stars:', a, 'Text:', b)
    #     print('---------------------')
    #
    # print(len(false_neg), 'False Negatives: ')
    # stars = [x['stars'] for x in false_neg]
    # text = [x['text'] for x in false_neg]
    # for a, b in zip(stars, text):
    #     print('Stars:', a, 'Text:', b)
    #     print('---------------------')


if __name__ == '__main__':
    main()

'''
# TESTING
training_data_file = 'training_data/yelp_training_sample_10000.json'
output_file = 'trained_bayes_output/trained_model_10000.json'
test_data_file = 'test_data/yelp_test_sample_10000.json'
train_model(training_data_file, output_file)
accuracies, errors = test_model(output_file, test_data_file)
'''
'''
Times for guess_1 vs guess_2 function
I did not run on training or testing data, instead just used trained_bayes_output files in function test_model
Timed on my laptop with time.clock(). Added results to make sure functions doing same thing.
output_file, test_data_file       guess_1 time:   guess_1_results:      guess_2 time:      guess_2_results:
2,2                                 .046875       (1.0, 1.0, 1)          .09375            (1.0, 1.0, 1)
2,50                                 .109         (.85106, 1.0, 0.0)     .078125           (.86106.., 1.0, 0.0)
50,2                                .046875       (1.0, 1.0, 1)          .0625             (1.0, 1.0, 1)
50,50                               .15625        (.7234, .825, .142)    .21875            (.7234, .825, .142..)
1500,2                              .09375        (1.0, 1.0, 1)          .09375            (1.0, 1.0, 1)
1500,50                             .375           (.74, .75, .71)       .265              (.74, .75, .7142)
500, 500                             2.8125       (.7042, .7329, .613)    2.453125         (.704, .7329, .613)
1500, 500                            3.6875                               3.84375
1000, 10000                          6.4375       (.73, .71, .79)        5.078125          (.732, .714, .79)
1500, 1000                          7.06           (.726, .71, .76)      5.453125          (.726, .71, .76)
'''
'''
accuracy_total, accuracy_pos, accuracy_neg = accuracies
false_pos, false_neg = errors

print('Total Accuracy: ', accuracy_total)
print('Positive Accuracy: ', accuracy_pos)
print('Negative Accuracy: ', accuracy_neg)
print('---------------------')

# print(len(false_pos), 'False Positives: ')
# stars = [x['stars'] for x in false_pos]
# text = [x['text'] for x in false_pos]
# for a, b in zip(stars, text):
#     print('Stars:', a, 'Text:', b)
#     print('---------------------')
#
# print(len(false_neg), 'False Negatives: ')
# stars = [x['stars'] for x in false_neg]
# text = [x['text'] for x in false_neg]
# for a, b in zip(stars, text):
#     print('Stars:', a, 'Text:', b)
#     print('---------------------')
'''
