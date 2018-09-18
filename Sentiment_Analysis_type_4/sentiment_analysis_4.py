# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 22:02:13 2018

@author: Nikhil
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud


datain = pd.read_csv("Sentiment.csv")

datain = datain[["text","sentiment"]]

from sklearn.model_selection import train_test_split
train, test = train_test_split(datain , test_size = 0.1, random_state = 0)

train = train[train.sentiment!="Neutral"]

postive_train = train[train['sentiment']=='Positive']
postive_train = postive_train['text']

negative_train = train[train['sentiment']=='Negative']
negative_train = negative_train['text']

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud().generate(cleaned_word)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

print("Positive words")
wordcloud_draw(postive_train,'white')

print("Negative words")
wordcloud_draw(negative_train)
                
tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.sentiment))

test_pos = test[ test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[ test['sentiment'] == 'Negative']
test_neg = test_neg['text']

def get_words_in_tweets(tweets):
    all_1 = []
    for (words, sentiment) in tweets:
        all_1.extend(words)
    return all_1

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features

w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

wordcloud_draw(w_features)

training_set = nltk.classify.apply_features(extract_features,tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

neg_cnt = 0
pos_cnt = 0
for obj in test_neg: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Negative'): 
        neg_cnt = neg_cnt + 1
for obj in test_pos: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Positive'): 
        pos_cnt = pos_cnt + 1
        
print('[Negative]: %s/%s '  % (neg_cnt,len(test_neg)))    
print('[Positive]: %s/%s '  % (pos_cnt, len(test_pos)))
