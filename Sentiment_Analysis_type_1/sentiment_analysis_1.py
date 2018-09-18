# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 23:02:50 2018

@author: Nikhil
"""

from nltk.classify import NaiveBayesClassifier
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

english_stopwords = stopwords.words('english')

waste_words = list(punctuation) + english_stopwords

def word_features(words):
    return dict([(words, True)])

 
positive_words = ['awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)']
negative_words = ['bad', 'terrible','useless', 'hate', 'worst',':(']
neutral_words = ['movie','the','sound','was','is','actors','did','know','words','not']
 
positive_features = [(word_features(pos), 'positive') for pos in positive_words]

negative_features = [(word_features(neg), 'negative') for neg in negative_words]

neutral_features = [(word_features(neu), 'neutral') for neu in neutral_words]
 
train_set = negative_features + positive_features + neutral_features

classifier = NaiveBayesClassifier.train(train_set) 
 
neg = 0
pos = 0
neu = 0
sentence = "movie is good, but sound quality is bad , worst , worst"
word_tokenize_array = word_tokenize(sentence)

words = [w for w in word_tokenize_array if w not in waste_words]

for word in words:
    classResult = classifier.classify(word_features(word))
    if classResult == 'positive':
        pos = pos + 1
    if classResult == 'negative':
        neg = neg + 1
    if classResult == 'neutral':
        neu = neu + 1
 
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))
print('Neutral: ' + str(float(neu)/len(words)))

print("\n##################################\n")

pos_1 = 0
neg_1 = 0
neu_1 = 0

for word in words:
    if word in positive_words:
        pos_1 = pos_1 + 1
    elif word in negative_words:
        neg_1 = neg_1 + 1
    else:
        neu_1 = neu_1 + 1

print('Positive: ' + str(float(pos_1)/len(words)))
print('Negative: ' + str(float(neg_1)/len(words)))
print('Neutral: ' + str(float(neu_1)/len(words)))
