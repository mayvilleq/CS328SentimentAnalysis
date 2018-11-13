'''
test.py contains functions to help with testing our three models, printing
results to a file, and analyzing errors.
'''

from naive_bayes import test_model as test_model_bayes
from naive_bayes import load_trained_output as load_output_bayes
from dictionary_model import load_trained_output as load_output_dict
from naive_bayes import normalize_word
from dictionary_model import test_model as test_model_dict
from seed_dictionaries import stop_words
import json


def test(trained_output_files_list, test_data_filename, files_to_write, indices_for_error_analysis):
    '''
    Tests all three models on the reviews in test_data_filename.

    trained_output_files_list is a 2-D list of the trained output files for each
    model in the order [naive_bayes, conjunction_dictionary, coccurrence_dictionary]
    where each inner list is a list of trained output files associated with that model
    varying with size of training data. We will assume each of these inner lists
    is of the same length and represents the same size order (increasing).
    files_to_write is a list of file names to which we will write our results. We will
    have one file name per each training size.

    We test for accuracy at each training size, but test for error analysis only at
    sizes in indices_for_error_analysis.

    We return a 2-D list of accuracies, where each list represents a training size,
    and the entries in the inner lists are accuracies tuples in the order [bayes, conjunction, co-occurrence]
    '''

    accuracies_list = []

    # Loop through all training data sizes - since all inner lists same size, can choose arbitrary one to determine length
    for i in range(len(trained_output_files_list[0])):
        file_to_write = files_to_write[i] #Write one file per size

        #If i not in indices_for_error_analysis, we just test accuracy
        if i not in indices_for_error_analysis:
            accuracies_bayes, errors_bayes = test_bayes(trained_output_files_list[0][i], test_data_filename, file_to_write, 'accuracies')
            accuracies_conj, errors_conj = test_dictionary_conj(trained_output_files_list[1][i], test_data_filename, file_to_write, 'accuracies')
            accuracies_co, errors_co = test_dictionary_co(trained_output_files_list[2][i], test_data_filename, file_to_write, 'accuracies')

        #Otherwise, test for accuracy and error analysis
        else:
            accuracies_bayes, errors_bayes = test_bayes(trained_output_files_list[0][i], test_data_filename, file_to_write, 'errors')
            accuracies_conj,errors_conj = test_dictionary_conj(trained_output_files_list[1][i], test_data_filename, file_to_write, 'errors')
            accuracies_co, errors_co = test_dictionary_co(trained_output_files_list[2][i], test_data_filename, file_to_write, 'errors')
            compare_across_models_print([trained_output_files_list[0][i], trained_output_files_list[1][i], trained_output_files_list[2][i]], errors_bayes, errors_conj, errors_co, test_data_filename, file_to_write)
        accuracies_list.append([accuracies_bayes, accuracies_conj, accuracies_co])

    return accuracies_list

def test_bayes(trained_output_filename, test_data_filename, file_to_write, type_of_test):
    '''
    Helper function for test(). Takes in a trained output file, testing data file,
    and an output file. It calls test_model from naive_bayes.py to determine the accuracies
    and errors (false_positives, false_negatives). It then appends accuracies
    (through print_result) to file_to_write if the type_of_test is 'accuracies' or both accuracies
    and errors if the type of test is 'errors.' Returns accuracies and errors.
    '''
    accuracies, errors = test_model_bayes(trained_output_filename, test_data_filename)
    if type_of_test is 'errors':
        print_result(accuracies, file_to_write, "Naive Bayes", trained_output_filename)
        top_10_false_lists = analyze_false_categorizations_bayes(errors, trained_output_filename)
        print_error_analysis(errors, top_10_false_lists, file_to_write, "Naive Bayes", trained_output_filename)
    else:
        print_result(accuracies, file_to_write, "Naive Bayes", trained_output_filename)

    return accuracies, errors

def test_dictionary_conj(trained_output_filename, test_data_filename, file_to_write, type_of_test):
    '''
    Helper function for test(). Takes in a trained output file, testing data file,
    and an output file. It calls test_model from dictionary_model.py to determine the accuracies
    and errors (false_positives, false_negatives). It then appends accuracies
    (through print_result) to file_to_write if the type_of_test is 'accuracies' or both accuracies
    and errors if the type of test is 'errors.' Returns accuracies and errors.
    '''
    accuracies, errors = test_model_dict(trained_output_filename, test_data_filename)
    if type_of_test is 'errors':
        print_result(accuracies, file_to_write, "Dictionary Model: Conjunction", trained_output_filename)
        top_10_false_lists = analyze_false_categorizations_dict(errors, trained_output_filename)
        print_error_analysis(errors, top_10_false_lists, file_to_write, "Dictionary Model: Conjunction", trained_output_filename)
    else:
        print_result(accuracies, file_to_write, "Dictionary Model: Conjunction", trained_output_filename)

    return accuracies, errors

def test_dictionary_co(trained_output_filename, test_data_filename, file_to_write, type_of_test):
    '''
    Helper function for test(). Takes in a trained output file, testing data file,
    and an output file. It calls test_model from dictionary_model.py to determine the accuracies
    and errors (false_positives, false_negatives). It then appends accuracies
    (through print_result) to file_to_write if the type_of_test is 'accuracies' or both accuracies
    and errors if the type of test is 'errors.' Returns accuracies and errors.
    '''
    accuracies, errors = test_model_dict(trained_output_filename, test_data_filename)
    if type_of_test is 'errors':
        print_result(accuracies,  file_to_write, "Dictionary Model: Co-occurrence", trained_output_filename)
        top_10_false_lists = analyze_false_categorizations_dict(errors, trained_output_filename)
        print_error_analysis(errors, top_10_false_lists, file_to_write, "Dictionary Model: Co-occurrence", trained_output_filename)
    else:
        print_result(accuracies,  file_to_write, "Dictionary Model: Co-occurrence", trained_output_filename)
    return accuracies, errors

def print_result(accuracies, file_to_write, model_title, trained_output_filename):
    '''
    Appends accuracies to trained_output_filename. Accuracies are taken
    in as parameters. Since we will use this function for multiple models, and
    print to the same output file, we take in a string model_title to print before
    writing results. We also print the trained_output_filename so we know the
    training size.
    '''
    # Unload accuracies
    accuracy_total, accuracy_pos, accuracy_neg = accuracies

    # Append results to file_to_write
    f = open(file_to_write, 'a')
    f.write(model_title + " with training file: " + trained_output_filename + "\n")
    f.write("Total Accuracy: " + str(accuracy_total) + "\n")
    f.write("Positive Accuracy: " + str(accuracy_pos) + "\n")
    f.write("Negative Accuracy: " + str(accuracy_neg) + "\n")
    f.write("\n \n ")


def analyze_false_categorizations_bayes(errors, trained_output_filename):
    '''
    This function takes in errors, a tuple of two lists of reviews: false negatives, false_positives.
    It then goes through the list of reviews in each false positives, to see which
    words in the reviews (other than words in stop words) have a large (positive likelihood - negative
    likelihood). We do the analagous task for the false negatives, and then return the top 10 words
    for both categories. The idea is to see why bayes classified these reviews incorrectly, i.e.
    which words are especially confusing to the model.
    '''

    # unload parameters
    (false_pos, false_neg) = errors
    word_list, priors, likelihoods = load_output_bayes(trained_output_filename)
    (pos_likelihood, neg_likelihood) = likelihoods

    # pos_likelihood_dict will store words as keys with (positive likelihood  - negative likelihood)
    # as the value. Similar for neg_likelihood_dict
    pos_likelihood_dict, neg_likelihood_dict = {}, {}

    # Iterate through false positives reviews and add to dictionary
    for review in false_pos:
        words = review["text"].split()
        for word in words:
            word = normalize_word(word)
            # stop words are a list in seed_dictionaries.py of words that we want to avoid: i.e., 'the', 'of'
            if word in word_list and word not in stop_words:
                word_index = word_list.index(word)
                if pos_likelihood_dict.get(word) is None:
                    pos_likelihood_dict[word] = pos_likelihood[word_index] - neg_likelihood[word_index]

    # Do same for false negatives
    for review in false_neg:
        # Look for words with high negative likelihood
        words = review["text"].split()
        for word in words:
            word = normalize_word(word)
            if word in word_list and word not in stop_words:
                word_index = word_list.index(word)
                if neg_likelihood_dict.get(word) is None:
                    neg_likelihood_dict[word] = neg_likelihood[word_index] - pos_likelihood[word_index]

    # Get top 10 confusing words for both cases
    pos_likelihood_sorted = sorted(pos_likelihood_dict, key=pos_likelihood_dict.get)
    neg_likelihood_sorted = sorted(neg_likelihood_dict, key=neg_likelihood_dict.get)
    top_10_false_pos = pos_likelihood_sorted[-20:]
    top_10_false_neg = neg_likelihood_sorted[-20:]

    return (top_10_false_pos, top_10_false_neg)


def analyze_false_categorizations_dict(errors, trained_output_filename):
    '''
    This function performs the analagous task of analyze_false_categorizations_bayes,
    but for dictionaries instead, taking in the same parameters. It goes through
    each review in false positives and counts how many times words in the positive
    dictionary (but not the negative dictionary, or stop words) appear and returns
    the 10 of these words that appear most often. It performs the analagous function
    for false negatives. The idea is to find the words confusing the models.
    '''

    # unload parameters
    false_pos, false_neg = errors
    positive, negative = load_output_dict(trained_output_filename)

    # pos_count_dict will store words in false positive reviews that are in
    # the positive dictionary, but not the negative dictionary. The values of these
    # words will be their count. Similar for neg_count_dict
    pos_count_dict = {}
    neg_count_dict = {}

    # iterate through false_positives
    for review in false_pos:
        words = review["text"].split()
        for word in words:
            word = normalize_word(word)
            if word in positive and word not in stop_words and word not in negative:
                if pos_count_dict.get(word) is None:
                    pos_count_dict[word] = 1
                else:
                    pos_count_dict[word] += 1

    # iterate through false_negatives
    for review in false_neg:
        words = review["text"].split()
        for word in words:
            word = normalize_word(word)
            if word in negative and word not in stop_words and word not in positive:
                if neg_count_dict.get(word) is None:
                    neg_count_dict[word] = 1
                else:
                    neg_count_dict[word] += 1

    # sort dictionaries, and get top 10 confusion words
    pos_counts_sorted = sorted(pos_count_dict, key=pos_count_dict.get)
    neg_counts_sorted = sorted(neg_count_dict, key=neg_count_dict.get)
    top_10_false_pos = pos_counts_sorted[-20:]
    top_10_false_neg = neg_counts_sorted[-20:]

    return (top_10_false_pos, top_10_false_neg)


def compare_across_models_print(trained_output_files_list, errors_bayes, errors_conj, errors_co, test_data_filename, file_to_write):
    '''
    Writes to file_to_write the size of shared false positives and shared false negatives
    between each subset of the three models.
    '''

    # Calls helper function to get overlap sizes
    shared_false_pos, shared_false_neg = compare_models_correctness(test_data_filename, errors_bayes, errors_conj, errors_co)

    # Write overlap to file_to_write
    f = open(file_to_write, 'a')
    f.write("Comparing Across Models with training files: " + trained_output_files_list[0] + ", " + trained_output_files_list[1] + ", " + trained_output_files_list[2] + "\n")
    f.write("Number of False Positives Misclassified by: \n")
    f.write("Bayes + Conj only: " + str(len(shared_false_pos["bayes_conj"])) + "\n")
    f.write("Bayes + Co-occurr only: " + str(len(shared_false_pos["bayes_co"])) + "\n")
    f.write("Co-occurr + Conj only: " + str(len(shared_false_pos["conj_co"])) + "\n")
    f.write("All three: " + str(len(shared_false_pos["bayes_conj_co"])) + "\n")
    f.write("-------------------------------------------------" + "\n")
    f.write("Number of False Negatives Misclassified by: " + "\n")
    f.write("Bayes + Conj only: " + str(len(shared_false_neg["bayes_conj"])) + "\n")
    f.write("Bayes + Co-occurr only: " + str(len(shared_false_neg["bayes_co"])) + "\n")
    f.write("Co-occurr + Conj only: " + str(len(shared_false_neg["conj_co"])) + "\n")
    f.write("All three: " + str(len(shared_false_neg["bayes_conj_co"])) + "\n \n")


def compare_models_correctness(test_data_filename, errors_bayes, errors_conj, errors_co):
    '''
    Returns two dictionaries, shared_false_pos and shared_false_neg which have keys
    of subsets of the three models, and values are the number of false positives (or negatives)
    that are classified incorrectly by models in that subset (and only by models in that subset).
    Takes in the test_data_filename so it can traverse through reviews in that file. Also takes
    in three tuples errors_model, containing false_positives, false_negatives for that model.
    '''

    # unload parameters
    false_pos_bayes, false_neg_bayes = errors_bayes
    false_pos_conj, false_neg_conj = errors_conj
    false_pos_co, false_neg_co = errors_co

    # get list of reviews to read through
    with open(test_data_filename) as test_data_file:
        reviews = json.loads(test_data_file.read())

    # set up dictionaries (described in function comment)
    shared_false_pos = {"bayes_conj":[], "bayes_co":[], "conj_co":[], "bayes_conj_co":[]}
    shared_false_neg = {"bayes_conj":[], "bayes_co":[], "conj_co":[], "bayes_conj_co":[]}

    # Iterate through each review in test data, adding to a category in a dictionary, if appropriate
    for review in reviews:

        # Check false positives
        if review in false_pos_bayes:
            if review in false_pos_conj:
                if review in false_pos_co:
                    shared_false_pos["bayes_conj_co"].append(review)
                else:
                    shared_false_pos["bayes_conj"].append(review)
            elif review in false_pos_co:
                shared_false_pos["bayes_co"].append(review)
        else:
            if review in false_pos_co and false_pos_conj:
                shared_false_pos["conj_co"].append(review)

        # Check false negatives
        if review in false_neg_bayes:
            if review in false_neg_conj:
                if review in false_neg_co:
                    shared_false_neg["bayes_conj_co"].append(review)
                else:
                    shared_false_neg["bayes_conj"].append(review)
            elif review in false_neg_co:
                shared_false_neg["bayes_co"].append(review)
        else:
            if review in false_neg_co and false_neg_conj:
                shared_false_neg["conj_co"].append(review)

    return shared_false_pos, shared_false_neg


def print_error_analysis(errors, top_10_false_lists, file_to_write, model_title, trained_output_filename):
    '''
    Appends error analysis to trained_output_filename. Since we will use this function
    for multiple models and print to the same output file, we take in a string
    model_title and trained_output_filename to print before writing analysis, for
    ease of writing. We print the first five false positives and false negatives
    (taken in as a tuple) in the parameter errors. We also print the top 10 words
    the confuse the model in false positives and false negatives.
    '''

    # unload parameters
    false_pos, false_neg = errors
    top_10_false_pos, top_10_false_neg = top_10_false_lists

    # Append results to file_to_write
    f = open(file_to_write, 'a')
    f.write("Error Analysis of " + model_title + " with training file: " + trained_output_filename + "\n")
    f.write("False Positives List: \n")
    f.write(json.dumps(false_pos[:5]))
    f.write("\n")
    f.write("False Negatives List: \n")
    f.write(json.dumps(false_neg[:5]))
    f.write("\n")
    f.write("Top 10 words contributing to false positives: \n")
    f.write(json.dumps(top_10_false_pos))
    f.write("\n")
    f.write("Top 10 words contributing to false negatives: \n")
    f.write(json.dumps(top_10_false_neg))
    f.write("\n \n ")


def main():
    trained_output_files_list = [["trained_bayes_output/trained_bayes_50.json", "trained_bayes_output/trained_bayes_100.json"], ["trained_conjunction_output/trained_conjunction_50.json", "trained_conjunction_output/trained_conjunction_100.json"], ["trained_cooccurrence_output/trained_cooccurrence_50.json", "trained_cooccurrence_output/trained_cooccurrence_100.json"]]
    files_to_write = ["testing_test_50.txt", "testing_test_100.txt"]
    test_data_filename = "test_data/yelp_test_sample_1000.json"

    test(trained_output_files_list, test_data_filename, files_to_write, [1])


# TODO remove main
if __name__ == '__main__':
    main()
