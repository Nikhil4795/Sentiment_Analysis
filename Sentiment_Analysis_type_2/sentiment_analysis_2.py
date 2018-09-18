# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 23:33:35 2018

@author: Nikhil
"""

## https://opensourceforu.com/2016/12/analysing-sentiments-nltk/

import nltk
from nltk.tokenize import word_tokenize
  
# Step 1 – Training data
train = [("Great place to be when you are in Bangalore.", "pos"),
  ("The place was being renovated when I visited so the seating was limited.", "neg"),
  ("Loved the ambience, loved the food", "pos"),
  ("The food is delicious but not over the top.", "neg"),
  ("Service - Little slow, probably because too many people.", "neg"),
  ("The place is not easy to locate", "neg"),
  ("Mushroom fried rice was spicy", "pos"),
]
  
dictionary = set(word.lower() for passage in train for word in word_tokenize(passage[0]))

all_words_dict = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]
  
classifier = nltk.NaiveBayesClassifier.train(all_words_dict)
  
test_data = "Manchurian was hot and spicy"
test_data_features = {word.lower(): (word in word_tokenize(test_data.lower())) for word in dictionary}
  
print(classifier.classify(test_data_features))
