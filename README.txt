# CS328SentimentAnalysis


DATA:

The complete Yelp data set is extremely large. It is available for download at https://www.yelp.com/dataset/challenge
The file needed is yelp_academic_dataset_review.json and it should be kept locally in a directory called yelp_dataset.

process_data.py contains code to randomly sample testing and training data sets from the complete data set.
Running process_data.random_sample_yelp_data(training_size, test_size) will write out these subsets to json files.
Running random_training_subset(filename, reviews_in_file, n) will take a random subset of size n of the given training file;
this function could be easily modified to take a random subset of testing (or any other) files.
The training and testing files we used are provided with the code.

The dictionary model requires some part of speech filtering for which we used the Natural Language Toolkit (NLTK).
Thus NLTK is required to run those models, as well as the NLTK data sets 'punkt' and 'averaged_perceptron_tagger'
which can be downloaded by the command nltk.download('punkt').
