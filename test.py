import naive_bayes
import dictionary_model
import json


#TODO do we still want test_model() in naive_bayes, or does this file replace it?
#(Have written as if it does, but can refactor)

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

        #TODO do we want these data loading functions in here, or keep in respective files?
        #Get necessary output for each  model
        word_list, prior, likelihood = naive_bayes.load_trained_output(trained_output_file_list[0])
        positive_conj, negative_conj = dictionary_model.load_trained_output(trained_output_file_list[1])
        positve_co, negative_co = dictionary_model.load_trained_output(trained_output_file_list[2])

        #Get test_data into list of reviews
        with open(test_data_filename) as test_data_file:
            reviews = json.loads(test_data_file.read())

        #Initialize false positives and false negatives list
        false_pos_bayes, false_neg_bayes = [], []  # list of reviews falsely categorized as positive/negative
        false_pos_conj, false_neg_conj = [], []
        false_pos_co, false_neg_co = [], []

        total_pos, total_neg = 0,0

        #Iterate through reviews, updating counts
        for review in reviews:
            sentiment, _, _ = get_sentiment_and_update_counts(review, 0, 0)
            if sentiment is 'n':
                continue

            bayes_guess = naive_bayes.guess_2_function(review, word_list, prior, likelihood)
            conj_guess = dictionary_model.guess(positive_conj, negative_conj, review)
            co_guess = dictionary_model.guess(positive_co, negative_co, review)

            if sentiment is '+':
                total_pos += 1
                if bayes_guess is '-':
                    false_neg_bayes.append(review)
                if conj_guess is '-':
                    false_neg_conj.append(review)
                if co_guess is '-':
                    false_neg_co.append(review)
            else:
                total_neg += 1
                if bayes_guess is '+':
                    false_pos_bayes.append(review)
                if conj_guess is '+':
                    false_pos_conj.append(review)
                if co_guess is '+':
                    false_pos_co.append(review)


        print_results(files_to_write[i], false_pos_bayes, false_neg_bayes, false_pos_conj, false_neg_conj, false_pos_co, false_neg_co, total_pos, total_neg)


def print_results(file_to_write, false_pos_bayes, false_neg_bayes, false_pos_conj, false_neg_conj, false_pos_co, false_neg_co, total_pos, total_neg):
    '''
    WRITE COMMENT HERE
    '''
    #TODO make this work for a variety of sizes....?

    correct_pos_bayes = total_pos - len(false_neg_bayes)
    correct_pos_conj= total_pos - len(false_neg_conj)
    correct_pos_co = total_pos - len(false_neg_co)

    correct_neg_bayes = total_neg - len(false_pos_bayes)
    correct_neg_conj= total_neg- len(false_pos_conj)
    correct_neg_co = total_neg- len(false_pos_co)

    if total_pos != 0:
        accuracy_pos_bayes = correct_pos_bayes / total_pos
        accuracy_pos_conj = correct_pos_conj / total_pos
        accuracy_pos_co = correct_pos_co / total_pos
    else:
        accuracy_pos_bayes, accuracy_pos_conj, accuracy_pos_co = 1, 1, 1  # TODO should this be default?

    if total_neg != 0:
        accuracy_neg_bayes = correct_neg_bayes / total_neg
        accuracy_neg_conj = correct_neg_conj / total_neg
        accuracy_neg_co = correct_neg_co / total_neg
    else:
        accuracy_neg_bayes, accuracy_neg_conj, accuracy_neg_co = 1, 1, 1


    accuracy_total_bayes = (correct_pos_bayes + correct_neg_bayes) / (total_pos + total_neg)
    accuracy_total_conj = (correct_pos_conj + correct_neg_conj) / (total_pos + total_neg)
    accuracy_total_co = (correct_pos_co + correct_neg_co) / (total_pos + total_neg)

    #write results to a file
    f = open(file_to_write, 'x')

    f.write("Naive Bayes Model \n" )
    f.write("Total Accuracy: " + str(accuracy_total_bayes) + "\n")
    f.write("Positive Accuracy: " + str(accuracy_pos_bayes) + " Negative Accuracy: " + str(accuracy_neg_bayes) + "\n")
    f.write("False Positives List: \n")    #TODO maybe we don't want to print these???
    f.write(json.dumps(false_pos_bayes))
    f.write("\n")
    f.write("False Negatives List: \n")
    f.write(json.dumps(false_neg_bayes))
    f.write("\n \n")

    f.write("Dictionary Model: Conjunction \n")
    f.write("Total Accuracy: " + str(accuracy_total_conj) + "\n")
    f.write("Positive Accuracy: " + str(accuracy_pos_conj) + " Negative Accuracy: " + str(accuracy_neg_conj) + "\n")
    f.write("False Positives List: \n")
    f.write(json.dumps(false_pos_conj))
    f.write("\n")
    f.write("False Negatives List: \n")
    f.write(json.dumps(false_neg_conj))
    f.write("\n \n")

    f.write("Dictionary Model: Cooccurence \n")
    f.write("Total Accuracy: " + str(accuracy_total_co) + "\n")
    f.write("Positive Accuracy: " + str(accuracy_pos_co) + " Negative Accuracy: " + str(accuracy_neg_co) + "\n")
    f.write("False Positives List: \n")
    f.write(json.dumps(false_pos_co))
    f.write("\n")
    f.write("False Negatives List: \n")
    f.write(json.dumps(false_neg_co))
    f.write("\n \n")


def get_sentiment_and_update_counts(review, num_pos, num_neg):
    #TODO also copied this from naive_bayes since it seems helpful...but Maybe
    #we should pass in (to not do things twice) or should just call it from naive_bayes
    '''
    Returns a tuple containing the sentiment of the given review and updated
    counts for the number of positive and negative reviews after processing
    the given review.
    '''
    sentiment = 'n'  # neutral
    num_stars = review["stars"]
    if num_stars >= 4:
        sentiment = '+'
        num_pos += 1
    elif num_stars <= 2:
        sentiment = '-'
        num_neg += 1
    return sentiment, num_pos, num_neg
