# CS328SentimentAnalysis


DATA:

The complete Yelp data set is extremely large. It is available for download at https://www.yelp.com/dataset/challenge
The file needed is yelp_academic_dataset_review.json and it should be kept locally in a directory called yelp_dataset.

process_data.py contains code to randomly sample testing and training data sets from the complete data set.
Running process_data.random_sample_yelp_data(training_size, test_size) will write out these subsets to json files.
Running random_training_subset(filename, reviews_in_file, n) will take a random subset of size n of the given training file;
this function could be easily modified to take a random subset of testing (or any other) files.

A subset of the training files we used are provided with the code in the training_data directory.
Additionally, the testing file we used is provided in the test_data directory as well as several
smaller random subsets of reviews that could be used for testing.


The dictionary model requires some part of speech filtering for which we used the Natural Language Toolkit (NLTK).
Thus NLTK is required to run those models, as well as the NLTK data sets 'punkt' and 'averaged_perceptron_tagger'
which can be downloaded by the command nltk.download('punkt').



TRAINING MODELS:

To train the Naive Bayes model, run naive_bayes.train_model(training_data_filename, output_filename)
where training_data_filename is the filename of the training data and output_filename is the name of
the file the trained model will write out to.
A subset of the output files of these trained models are included in the trained_bayes_output directory.


To train the Conjunction model, run dictionary_model.train_conjunction_model(training_data_filename, output_filename)
where training_data_filename is the filename of the training data and output_filename is the name of
the file the trained model will write out to.
A subset of the output files of these trained models are included in the trained_conjunction_output directory.


To train the Co-Occurrence model, run dictionary_model.train_cooccurrence_model(training_data_filename, output_filename)
where training_data_filename is the filename of the training data and output_filename is the name of
the file the trained model will write out to.
A subset of the output files of these trained models are included in the trained_cooccurrence_output directory.



TESTING MODELS:

To test a trained Naive Bayes model, run naive_bayes.test_model(trained_output_filename, test_data_filename)
where trained_output_filename is the filename of the output from train_model and test_data_filename is the name of
the filecontaining test data. This function will return a tuple of accuracies and errors. Where accuracies is a tuple of the
resulting total accuracy, positive accuracy, and negative accuracy, and errors is a tuple of false positives and false negatives

Similarly, to test any of the trained dictionary models, run dictionary_model.test_model(trained_output_filename, test_data_filename)
to get the accuracy, error output described above for either dictionary model.

To test all the models across many training sizes, run test.test(). The parameters for this function are
a bit complicated, but an explanation can be found in the comment under the function. Examples of how to
utilize this function can be found in the Jupyter notebook Result_and_Error_analysis.ipynb. Additionally,
samples of the output files from this function can are included in the test_results directory. The Jupyter
notebook also contains code used to construct graphs from the test results, and this code could be easily
modified to create new graphs. The test.py module contains several other functions that could be useful for testing.
