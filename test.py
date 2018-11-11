from naive_bayes import test_model as test_model_bayes
from naive_bayes import load_trained_output as load_output_bayes
from dictionary_model import load_trained_output as load_output_dict
from naive_bayes import normalize_word
from dictionary_model import test_model as test_model_dict
import json


def test(trained_output_files_list, test_data_filename, files_to_write):
    '''
    Tests all three models on the reviews in test_data_filename.

    trained_output_files_list is a 2-D array of the trained output files for each
    model in the order [naive_bayes, conjunction_dictionary, coccurrence_dictionary]
    where each inner list is a list of trained output files associated with that model
    varying with size of training data. We will assume each of these inner lists
    is of the same length.

    files_to_write is a list of file names - we will write results of all three
    methods to the same file, but will make a separate file for each sample size
    '''

    #Loop through all training data sizes - since all inner lists same size, can choose arbitrary one
    for i in range(len(trained_output_files_list[0])):
        file_to_write = files_to_write[i]
        test_bayes(trained_output_files_list[0][i], test_data_filename, file_to_write)
        test_dictionary_conj(trained_output_files_list[1][i], test_data_filename, file_to_write)
        test_dictionary_co(trained_output_files_list[2][i], test_data_filename, file_to_write)

def test_bayes(trained_output_filename, test_data_filename, file_to_write):
    '''
    WRITE COMMENT HERE
    '''
    accuracies, errors = test_model_bayes(trained_output_filename, test_data_filename)
    top_10_false_lists = [], []
    print_result(accuracies, errors, top_10_false_lists, file_to_write, "Naive Bayes", trained_output_filename)

def test_dictionary_conj(trained_output_filename, test_data_filename, file_to_write):
    '''
    WRITE COMMENT
    '''
    accuracies, errors = test_model_dict(trained_output_filename, test_data_filename)
    top_10_false_lists = analyze_false_categorizations_dict(errors, trained_output_filename)
    print_result(accuracies, errors, top_10_false_lists, file_to_write, "Dictionary Model: Conjunction", trained_output_filename)

def test_dictionary_co(trained_output_filename, test_data_filename, file_to_write):
    '''
    WRITE COMMENT
    '''
    accuracies, errors = test_model_dict(trained_output_filename, test_data_filename)
    top_10_false_lists = analyze_false_categorizations_dict(errors, trained_output_filename)
    print_result(accuracies, errors, top_10_false_lists, file_to_write, "Dictionary Model: Co-occurence", trained_output_filename)

def print_result(accuracies, errors, top_10_false_lists, file_to_write, model_title, trained_output_filename):
    '''
    WRITE COMMENT
    '''

    (accuracy_total, accuracy_pos, accuracy_neg) = accuracies
    (false_pos, false_neg) = errors
    (top_10_false_pos, top_10_false_neg) = top_10_false_lists

    #write results to a file TODO mention why append in comment and to Quinn
    f = open(file_to_write, 'a')
    f.write(model_title + "\n" )
    f.write("Total Accuracy: " + str(accuracy_total) + "\n")
    f.write("Positive Accuracy: " + str(accuracy_pos) + "\n")
    f.write(" Negative Accuracy: " + str(accuracy_neg) + "\n")
    f.write("False Positives List: \n")    #TODO maybe we don't want to print these - too long???
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
    WRITE COMMENT HERE
    '''
    (false_pos, false_neg) = errors


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
            if word in positive:
                if pos_count_dict.get(word) is None:
                    pos_count_dict[word] = 1
                else:
                    pos_count_dict[word] += 1
    for review in false_neg:
        # Look for words that contribute highly to negative count...
        words = review["text"].split()
        for word in words:
            word = normalize_word(word)
            if word in negative:
                if neg_count_dict.get(word) is None:
                    neg_count_dict[word] = 1
                else:
                    neg_count_dict[word] += 1
    pos_counts_sorted = sorted(pos_count_dict, key=pos_count_dict.get)
    neg_counts_sorted = sorted(neg_count_dict, key=neg_count_dict.get)
    top_10_false_pos = pos_counts_sorted[-10:]
    top_10_false_neg = neg_counts_sorted[-10:]
    print(top_10_false_pos, top_10_false_neg)
    return (top_10_false_pos, top_10_false_neg)

def compare_models_correctness(test_data_filename, false_pos_bayes, false_neg_bayes, false_pos_conj, false_neg_conj, false_pos_co, false_neg_co):

    with open(test_data_filename) as test_data_file:
        reviews = json.loads(test_data_file.read())
    #TODO: does it make sense to do separateley? use all's
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
                
def main():
    #TESTING
    trained_output_files_list = [["trained_bayes_output/trained_model_1000.json"], ["trained_dictionary_output/trained_conjunction_model_1000.json"], ["trained_dictionary_output/trained_cooccurrence_model_1000.json"]]
    files_to_write = ["size_1000_test_take_2.txt"]
    test_data_filename = "test_data/yelp_test_sample_2.json"
    test(trained_output_files_list, test_data_filename, files_to_write)

if __name__ == '__main__':
    main()
