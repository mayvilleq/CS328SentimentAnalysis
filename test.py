import naive_bayes
import dictionary_model

def test(trained_output_file_list, test_data_filename, file_to_write):
    #TODO do we still want test_model() in naive_bayes
    '''
    Tests all three models on the reivews in test_data_filename. Writes results
    to file_to_write. trained_output_file_list is a list of the trained output
    files for each model in the order [naive_bayes, conjunction_dictionary, coccurrence_dictionary]
    Modeled after test_model() in naive_bayes.py
    '''

    #TODO do we want these data loading files in here, or keep in respective files?
    #Output data for naive_bayes
    word_list, prior, likelihood = naive_bayes.load_trained_output(trained_output_file_list[0])

    #Output data for dictionary approaches
    positive_conj, negative_conj = dictionary_model.load_trained_output(trained_output_file_list[1])
    positve_cooccur, negative_cooccur = dictionary_model.load_trained_output(trained_output_file_list[2])

    #Get test_data into list of reviews
    with open(test_data_filename) as test_data_file:
        reviews = json.loads(test_data_file.read())
    for review in reviews:
        continue
        #TODO write this part....
