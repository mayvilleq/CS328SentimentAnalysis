'''
test.py contains functions to help with testing our three models, printing
results, and analyzing errors.
'''

from naive_bayes import test_model as test_model_bayes
from naive_bayes import load_trained_output as load_output_bayes
from dictionary_model import load_trained_output as load_output_dict
from naive_bayes import normalize_word
from dictionary_model import test_model as test_model_dict
from seed_dictionaries import (
    negative_seeds, positive_seeds, stop_words
)
import json


def test(trained_output_files_list, test_data_filename, files_to_write):
    '''
    Tests all three models on the reviews in test_data_filename.

    trained_output_files_list is a 2-D list of the trained output files for each
    model in the order [naive_bayes, conjunction_dictionary, coccurrence_dictionary]
    where each inner list is a list of trained output files associated with that model
    varying with size of training data. We will assume each of these inner lists
    is of the same length and represents the same size order (increasing).

    files_to_write is a list of file names: we will write results of all three
    methods to the same file, but will make a separate file for each sample size
    '''
    #Loop through all training data sizes - since all inner lists same size, can choose arbitrary one to determine length
    for i in range(len(trained_output_files_list[0])):
        file_to_write = files_to_write[i]
        test_bayes(trained_output_files_list[0][i], test_data_filename, file_to_write)
        test_dictionary_conj(trained_output_files_list[1][i], test_data_filename, file_to_write)
        test_dictionary_co(trained_output_files_list[2][i], test_data_filename, file_to_write)

def test_bayes(trained_output_filename, test_data_filename, file_to_write):
    '''
    Helper function for test(). Takes in a trained output file, testing data file,
    and an output file. It calls test_model from naive_bayes.py to determine the accuracies
    and errors (false_positives, false_negatives). It then appends these results
    (through print_result) to file_to_write.
    '''
    accuracies, errors = test_model_bayes(trained_output_filename, test_data_filename)
    top_10_false_lists = analyze_false_categorizations_bayes(errors, trained_output_filename)
    print_result(accuracies, errors, top_10_false_lists, file_to_write, "Naive Bayes", trained_output_filename)

def test_dictionary_conj(trained_output_filename, test_data_filename, file_to_write):
    '''
    Helper function for test(). Takes in a trained output file, testing data file,
    and an output file. It calls test_model from dictionary_model.py to determine the accuracies
    and errors (false_positives, false_negatives). It then appends these results
    (through print_result) to file_to_write.
    '''
    accuracies, errors = test_model_dict(trained_output_filename, test_data_filename)
    top_10_false_lists = analyze_false_categorizations_dict(errors, trained_output_filename)
    print_result(accuracies, errors, top_10_false_lists, file_to_write, "Dictionary Model: Conjunction", trained_output_filename)

def test_dictionary_co(trained_output_filename, test_data_filename, file_to_write):
    '''
    Helper function for test(). Takes in a trained output file, testing data file,
    and an output file. It calls test_model from dictionary_model.py to determine the accuracies
    and errors (false_positives, false_negatives). It then appends these results
    (through print_result) to file_to_write.
    '''
    accuracies, errors = test_model_dict(trained_output_filename, test_data_filename)
    top_10_false_lists = analyze_false_categorizations_dict(errors, trained_output_filename)
    print_result(accuracies, errors, top_10_false_lists, file_to_write, "Dictionary Model: Co-occurence", trained_output_filename)

def print_result(accuracies, errors, top_10_false_lists, file_to_write, model_title, trained_output_filename):
    '''
    Appends results to trained_output_filename. Results (accuracies/errors) are taken
    in as parameters. Since we will use this function for multiple models, and
    print to the same output file, we take in a string model_title to print before
    writing results. At the moment, we also take in
    '''
    #TODO decide whether you want to deal with accuracies and errors here...or make separate function - add to comment to tell what you decide
    # Unload tuples passed in
    (accuracy_total, accuracy_pos, accuracy_neg) = accuracies
    (false_pos, false_neg) = errors
    (top_10_false_pos, top_10_false_neg) = top_10_false_lists

    # Append results to file_to_write
    f = open(file_to_write, 'a')
    f.write(model_title + "\n" )
    f.write("Total Accuracy: " + str(accuracy_total) + "\n")
    f.write("Positive Accuracy: " + str(accuracy_pos) + "\n")
    f.write(" Negative Accuracy: " + str(accuracy_neg) + "\n")
    f.write("False Positives List: \n")
    f.write(json.dumps(false_pos))
    f.write("\n")
    f.write("False Negatives List: \n")
    f.write(json.dumps(false_neg))
    f.write("\n")
    f.write("Top 10 words contributing to false positives: \n")
    f.write(json.dumps(top_10_false_pos))
    f.write("\n")
    f.write("Top 10 words contributing to false negatives: \n")
    f.write(json.dumps(top_10_false_neg))
    f.write("\n \n ")

def analyze_false_categorizations_bayes(errors, trained_output_filename):
    '''
    This function takes in errors, a tuple of two lists of reviews: false negatives, false_positives.
    It then goes through the list of reviews in each false positives, to see which
    words in the reviews (other than words in stop words) a high likelihood p(w| + ). We do
    the analagous task for the false negatives, and then return the top 10 words
    for both categories based on likelihood. The idea is to see why bayes classified
    these reviews incorrectly, i.e. which words are especially confusing to the model.
    '''
    #TODO keep commenting here and rest of document. Left off here
    (false_pos, false_neg) = errors
    word_list, priors, likelihoods = load_output_bayes(trained_output_filename)
    (pos_likelihood, neg_likelihood) = likelihoods
    pos_likelihood_dict, neg_likelihood_dict = {}, {}
    for review in false_pos:
        #Look for words with high positive likelihood
        words = review["text"].split()
        for word in words:
            word = normalize_word(word)
            if word in word_list and word not in stop_words:
                word_index = word_list.index(word)
                if pos_likelihood_dict.get(word) is None:
                    pos_likelihood_dict[word] = pos_likelihood[word_index] - neg_likelihood[word_index]

                #if pos_likelihood_dict.get(word) is None:
                #    pos_likelihood_dict[word] = pos_likelihood[word_index]
                #else:
                #    pos_likelihood_dict[word] *= pos_likelihood[word_index]
    for review in false_neg:
        #Look for words with high negative likelihood
        words = review["text"].split()
        for word in words:
            word = normalize_word(word)
            if word in word_list and word not in stop_words:
                word_index = word_list.index(word)
                if neg_likelihood_dict.get(word) is None:
                    neg_likelihood_dict[word] = neg_likelihood[word_index] - pos_likelihood[word_index]
                #else:
                #    neg_likelihood_dict[word] *= neg_likelihood[word_index]

    pos_likelihood_sorted = sorted(pos_likelihood_dict, key=pos_likelihood_dict.get)
    neg_likelihood_sorted = sorted(neg_likelihood_dict, key=neg_likelihood_dict.get)
    top_10_false_pos = pos_likelihood_sorted[-20:]
    top_10_false_neg = neg_likelihood_sorted[-20:]
    print(top_10_false_pos, top_10_false_neg)
    return (top_10_false_pos, top_10_false_neg)



def analyze_false_categorizations_dict(errors, trained_output_filename):
    '''returns words in false_pos that contribute the most positive weight'''

    #TODO: actually test this on lab computers....first make sure dictionary.py okay...
    (false_pos, false_neg) = errors
    positive, negative = load_output_dict(trained_output_filename)
    pos_count_dict = {}
    neg_count_dict = {}
    for review in false_pos:
        #Look for words that contribute to positive count...
        words = review["text"].split()
        for word in words:
            word = normalize_word(word)
            if word in positive and word not in stop_words:
                if pos_count_dict.get(word) is None:
                    pos_count_dict[word] = 1
                else:
                    pos_count_dict[word] += 1
    for review in false_neg:
        # Look for words that contribute highly to negative count...
        words = review["text"].split()
        for word in words:
            word = normalize_word(word)
            if word in negative and word not in stop_words:
                if neg_count_dict.get(word) is None:
                    neg_count_dict[word] = 1
                else:
                    neg_count_dict[word] += 1
    pos_counts_sorted = sorted(pos_count_dict, key=pos_count_dict.get)
    neg_counts_sorted = sorted(neg_count_dict, key=neg_count_dict.get)
    top_10_false_pos = pos_counts_sorted[-20:]
    top_10_false_neg = neg_counts_sorted[-20:]
    print(top_10_false_pos, top_10_false_neg)
    return (top_10_false_pos, top_10_false_neg)

def compare_across_models(trained_output_files_list, test_data_filename):
    '''
    WRITE COMMENT HERE
    '''
    errors_bayes = test_model_bayes(trained_output_files_list[0], test_data_filename)[1]
    errors_conj = test_model_dict(trained_output_files_list[1], test_data_filename)[1]
    errors_co = test_model_dict(trained_output_files_list[2], test_data_filename)[1]

    shared_false_pos, shared_false_neg = compare_models_correctness(test_data_filename, errors_bayes, errors_conj, errors_co)
    print("Number of False Positives Misclassified by: ")
    print("Bayes + Conj only: ", len(shared_false_pos["bayes_conj"]))
    print("Bayes + Co-occurr only: ", len(shared_false_pos["bayes_co"]))
    print("Co-occurr + Conj only: ", len(shared_false_pos["conj_co"]))
    print("All three: ", len(shared_false_pos["bayes_conj_co"]))
    print("-------------------------------------------------")
    print("Number of False Negatives Misclassified by: ")
    print("Bayes + Conj only: ", len(shared_false_neg["bayes_conj"]))
    print("Bayes + Co-occurr only: ", len(shared_false_neg["bayes_co"]))
    print("Co-occurr + Conj only: ", len(shared_false_neg["conj_co"]))
    print("All three: ", len(shared_false_neg["bayes_conj_co"]))


def compare_models_correctness(test_data_filename, errors_bayes, errors_conj, errors_co):
    '''
    WRITE COMMENT HERE
    '''
    #TODO use set operations instead
    (false_pos_bayes, false_neg_bayes) = errors_bayes
    (false_pos_conj, false_neg_conj) = errors_conj
    (false_pos_co, false_neg_co) = errors_co
    with open(test_data_filename) as test_data_file:
        reviews = json.loads(test_data_file.read())
    print(len(reviews))
    shared_false_pos = {"bayes_conj":[], "bayes_co":[], "conj_co":[], "bayes_conj_co":[]}
    shared_false_neg = {"bayes_conj":[], "bayes_co":[], "conj_co":[], "bayes_conj_co":[]}
    for review in reviews:
        # False positives first:
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

        # Now false negatives:
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


def main():
    #TESTING
    trained_output_files_list = ["trained_bayes_output/trained_model_1000.json", "trained_dictionary_output/trained_conjunction_model_1000.json", "trained_dictionary_output/trained_cooccurrence_model_1000.json"]
    files_to_write = ["testing_test.txt"]
    test_data_filename = "test_data/yelp_test_sample_10000.json"
    #test(trained_output_files_list, test_data_filename, files_to_write)
    compare_across_models(trained_output_files_list, test_data_filename)
if __name__ == '__main__':
    main()
