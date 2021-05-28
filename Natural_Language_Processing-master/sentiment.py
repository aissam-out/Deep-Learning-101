import os
import re
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

# load the train and test data
# The labeled data set consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis.
# The sentiment of reviews is binary, (i.e. IMDB rating < 5 results in a sentiment score of 0. IMDB >=7 have a score of 1.
# There are 25,000 review labeled training set & 25,000 review test set & 50,000 IMDB reviews provided without any rating labels.
train = pd.read_csv("./data/labeledTrainData.tsv", header = 0, delimiter = '\t')
test = pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", quoting=3 )

# To clean the texts
def review_to_words(raw_review):
        review_text = BeautifulSoup(raw_review, features="html5lib").get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
        return(" ".join( meaningful_words ))

num_reviews = train["review"].size
clean_train_reviews = []

for i in range(0, num_reviews):
        clean_train_reviews.append(review_to_words(train["review"][i]))

# Creating the Bag of Words model
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

# Random Forest classifier
forest = RandomForestClassifier(n_estimators = 150)
forest = forest.fit(train_data_features, train["sentiment"])

num_reviews = len(test["review"])
clean_test_reviews = []

for i in range(0,num_reviews):
    clean_review = review_to_words(test["review"][i] )
    clean_test_reviews.append(clean_review )

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Predicting the test set results
result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Save the results
output.to_csv("submit.csv", index=False, quoting=3 )
