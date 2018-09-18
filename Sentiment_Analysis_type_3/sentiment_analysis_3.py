# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 16:12:28 2018

@author: Nikhil
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer 

hotel_reviews = ["Great place to be when you are in Bangalore.",\
             "The place was being renovated when I visited so the seating was limited.",\
             "Loved the ambience, loved the food",\
             "The food is delicious but not over the top.",\
             "Service - Little slow, probably because too many people.",\
             "The place is not easy to locate",\
             "Mushroom fried rice was tasty",\
             "Manchurian was hot and spicy"]

 
sentment_analyzer = SentimentIntensityAnalyzer()

for sentence in hotel_reviews:
     print(sentence)
     scores = sentment_analyzer.polarity_scores(sentence)
     for each in scores:
         print("{0}: {1}, ".format(each, scores[each]), end='')
     print("\n")
