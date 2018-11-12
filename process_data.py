import json
import random


# TODO deal with encoding (/u00...)
def random_sample_yelp_data(training_size, test_size):
    '''
    Randomly samples the Yelp review data set into a training subset and a test
    subset, both of which contain no neutral reviews. Writes out the data sets into new files.

    Note that run time can be very long. Infrequently fails due to too many neutral
    reviews. If it fails, just try again.
    '''
    training_data = []
    test_data = []
    random_sample = random.sample(range(1, 5996996), 2*(training_size + test_size))  # 5,996,996 total lines
    random.shuffle(random_sample)

    with open('yelp_dataset/yelp_academic_dataset_review.json') as data_file:
        i = 1
        pos_train, neg_train, pos_test, neg_test = 0, 0, 0, 0
        for line in data_file:
            if len(training_data) == training_size and len(test_data) == test_size:
                break

            review = json.loads(line)
            stars = int(review['stars'])
            if stars == 3:
                continue

            if i in random_sample and len(training_data) < training_size:
                training_data.append(review)
                if stars > 3:
                    pos_train += 1
                else:
                    neg_train += 1
            elif i in random_sample and len(test_data) < test_size:
                test_data.append(review)
                if stars > 3:
                    pos_test += 1
                else:
                    neg_test += 1

            i += 1

    test_filename = 'test_data/yelp_test_sample_{t}total_{p}pos_{n}neg.json'.format(t=test_size, p=pos_test, n=neg_test)
    training_filename = 'training_data/yelp_training_sample_{t}total_{p}pos_{n}neg.json'.format(t=training_size, p=pos_train, n=neg_train)

    with open(test_filename, 'w') as outfile:
        json.dump(test_data, outfile)

    with open(training_filename, 'w') as outfile:
        json.dump(training_data, outfile)


def random_training_subset(filename, reviews_in_file, n):
    '''
    Randomly samples n reviews from the given file and writes out the subset
    to a new file.

    Sample Runtimes for different n values:
    2500: 533 seconds
    500: 114 seconds
    250: 62 seconds
    25: 8 seconds

    Roughly n/5 seconds
    '''
    training_data = []
    random_sample = random.sample(range(1, reviews_in_file), n)
    random.shuffle(random_sample)

    with open('yelp_dataset/yelp_academic_dataset_review.json') as data_file:
        i = 1
        size = 0
        pos, neg = 0, 0
        for line in data_file:
            if i in random_sample:
                review = json.loads(line)
                training_data.append(review)
                if int(review['stars']) > 3:
                    pos += 1
                else:
                    neg += 1
                size += 1
            if size == n:  # If done with sample, stop going through file
                break
            i += 1

    training_filename = 'training_data/yelp_training_sample_{t}total_{p}pos_{n}neg.json'.format(t=n, p=pos, n=neg)
    with open(training_filename, 'w') as outfile:
        json.dump(training_data, outfile)
