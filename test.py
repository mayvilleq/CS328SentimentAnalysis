import naive_bayes
import dictionary_model
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
    accuracies, errors = naive_bayes.test_model(trained_output_filename, test_data_filename)
    print_result(accuracies, errors, file_to_write, "Naive Bayes")

def test_dictionary_conj(trained_output_file, test_data_filename, file_to_write):
    ''' 
    WRITE COMMENT
    '''

    accuracies, errors = dictionary_model.test_model(trained_output_filename, test_data_filename)
    print_result(accuracies, errors, file_to_write, "Dictionary Model: Conjunction")

def test_dictionary_co(trained_output_file, test_data_filename, file_to_write):
    ''' 
    WRITE COMMENT
    '''
    accuracies, errors = dictionary_model.test_model(trained_output_filename, test_data_filename)
    print_result(accuracies, errors, file_to_write, "Dictionary Model: Co-occurence")

def print_result(accuracies, errors, file_to_write, model_title):
    '''
    WRITE COMMENT
    '''

    (accuracy_total, accuracy_pos, accuracy_neg) = accuracies
    (false_pos, false_neg) = errors

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
    f.write("\n \n")