import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility

import pandas as pd #pandas helps us read our data CSV files
import numpy as np
import nltk #remove unnecessary words from our dataset

def main():
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3) #se nao tivesse header era header=None
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )    #quoting=3 is to ignore all the quotes in the way

    #nltk.download()  # Download text data sets, including stop words
                     # this is essential to clean it that means ensure that we remove all the HTML, non letters and stop words
                     # stop words are words that are insignificant (words like "the" or "to" or "as" since it's hard to analyze emotion from them)


    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Cleaning and parsing the training set movie reviews.
    # Loop over each review; create an index i that goes from 0 to the length of the movie review list
    for i in range( 0, len(train["review"])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

    # ****** Create a bag of words from the training set
    # the bag of words model is a simple numeric representation of a piece of text that is easy to classify
    # we just count the frequency of each word in a piece of text and create a dictionary of them - TOKENIZATION

    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,         # this has also the possibility of cleaning the stop words
                             max_features = 5000)       # will contain at max 5000 words and their associated frequencies to keep the things simple

    # fit_transform() does two functions: First, it fits the model and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews) #we use a fit transform method to model to the bag of words and create the feature vectors

    # Numpy arrays are easy to work with, so convert the result to an array
    np.asarray(train_data_features)


    # ******* Train a random forest using the bag of words
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the forest to the training set, using the bag of words as features and the sentiment labels as the response variable
    forest = forest.fit(train_data_features, train["sentiment"]) #train["sentiment"] is our labeled data

    # Create an empty list and append the clean reviews one by one
    clean_test_reviews = []

    # Cleaning and parsing the test set movie reviews
    for i in range(0,len(test["review"])):
       clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

    # Now it is time to create a classifier. A classifier is a machine learning model that will be used to classify
    # whether a piece of text is positive or negative in this example.
    # Use the random forest to make sentiment label predictions
    # Predicting test labels
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

    # Use pandas to write the comma-separated output file
    output.to_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'Bag_of_Words_model.csv'), index=False, quoting=3)


if __name__ == "__main__":
    main()
