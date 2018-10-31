import json
import random


# TODO deal with encoding (/u00...)
def random_sample_yelp_data(n):
    '''
    Randomly samples the Yelp review data set into a training data set of size n
    and a test data set of size n. Writes out the data sets into new files.

    Sample Runtimes for different n values:
    2500: 533 seconds
    500: 114 seconds
    250: 62 seconds
    25: 8 seconds

    Roughly n/5 seconds
    '''
    training_data = []
    test_data = []
    random_sample = random.sample(range(1, 5996996), 2*n)  # 5,996,996 total lines
    random.shuffle(random_sample)
    training_sample = random_sample[:n]
    test_sample = random_sample[n:]

    with open('yelp_dataset/yelp_academic_dataset_review.json') as data_file:
        i = 1
        size = 0
        for line in data_file:
            if i in training_sample:
                line_data = json.loads(line)
                training_data.append(line_data)
                size += 1
            if i in test_sample:
                line_data = json.loads(line)
                test_data.append(line_data)
                size += 1
            if size == 2*n:  # If done with sample, stop going through file
                break
            i += 1

    with open('test_data/yelp_test_sample_{n}.json'.format(n=n), 'w') as outfile:
        json.dump(test_data, outfile)

    with open('training_data/yelp_training_sample_{n}.json'.format(n=n), 'w') as outfile:
        json.dump(training_data, outfile)
