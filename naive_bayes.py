import json
import string
# sample function on how to get data
def read_data_from_sample(training_data_filename, output_filename):
    with open(training_data_filename) as data_file:
        reviews = json.loads(data_file.read())
    num_positive = 0
    num_negative = 0
    sentiment_list = []
    word_list = []
    word_count_pos = []
    word_count_neg = []
    for review in reviews:
        sentiment, num_positive, num_negative = get_sentiment_and_update_counts(review, num_positive, num_negative)
        sentiment_list.append(sentiment)
        words = review["text"].split()
        for word in words:
            word = word.lower().strip()
            for char in string.punctuation:
                word = word.strip(char)
            if word is '':
                continue
            if word not in word_list:
                word_list.append(word)
                word_count_pos.append(0)
                word_count_neg.append(0)
                index = len(word_list) - 1
            else:
                index = word_list.index(word)
            if sentiment is '+':
                word_count_pos[index] += 1
            if sentiment is '-':
                word_count_neg[index] += 1
    prob_pos = num_positive / (num_positive + num_negative)
    prob_neg = num_negative / (num_positive + num_negative)
    prior = (prob_pos, prob_neg)
    #for i in range(len(word_list)):
        #print(word_list[i], ": ", word_count_pos[i])

    pos_likelihood = []
    neg_likelihood = []
    for i in range(len(word_list)):
        pos_likelihood.append(1 + word_count_pos[i])
        neg_likelihood.append(1 + word_count_neg[i])
    #Normalizing the likelihoods
    pos_likelihood = [x/sum(pos_likelihood) for x in pos_likelihood]
    neg_likelihood = [x/sum(neg_likelihood) for x in neg_likelihood]
    likelihood = (pos_likelihood, neg_likelihood)
    data_dict = {"word_list":word_list, "prior":prior, "likelihood":likelihood}
    with open(output_filename, 'w') as outfile:
        json.dump(data_dict, outfile)
    # for i, word in enumerate(word_list):
    #     print(word, ": ", pos_likelihood[i])
    # print(neg_likelihood)




def get_sentiment_and_update_counts(review, num_positive, num_negative):
    sentiment = 'n'    #neutral
    num_stars = review["stars"]
    if num_stars >= 4:
        sentiment = '+'
        num_positive += 1
    elif num_stars <= 2:
        sentiment = '-'
        num_negative += 1
    return sentiment, num_positive, num_negative

read_data_from_sample('training_data/yelp_training_sample_2.json', 'trained_bayes_output/test_2.json' )
