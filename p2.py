#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:00:01 2018

@author: srinidhi
"""
import csv
#import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Word2Vec

fil = list(csv.reader(open('mbti_1.csv')))
tknzr = TweetTokenizer()
sentAn = SentimentIntensityAnalyzer() #sentiment analyzer

stopWords = set(stopwords.words('english'))
rawtrain = [x[1] for x in fil[1:20]]
#print(rawtrain[2])
count = {}
train = {}
stopWordsFreq = {}
sentAnScores = {}
textTrack = {}

''' #Sentiment Analysis Test Code
ss = sentAn.polarity_scores(fil[1][1])
for k in ss:
    print('{0}: {1}, '.format(k, ss[k]), end='')
    print(type(k),type(ss[k]))
print()
'''

for j in fil[1:8000]:
    i = j[1]    
    temp = i.split('|||')
    if j[0] not in textTrack:
        textTrack[j[0]] = []
    for x in temp:
        if j[0] == 'INFP':
            textTrack[j[0]].append(list(tknzr.tokenize(x)))

#t1 = Word2Vec(textTrack['INFJ'], min_count = 1)
#t2 = Word2Vec(textTrack['ENFJ'].sents())
#for i in textTrack:
i='INFP'
model = Word2Vec(textTrack[i], min_count = 1)
words = list(model.wv.vocab)
print(words)
result = model.most_similar(negative=['should'])#,topn=20)
print(result)
print('')
result = model.most_similar(negative=['i'])#,topn=20)
print(result)
print('')
result = model.most_similar(negative=['think'],topn=20)
print(result)
#    print(t1.most_similar('should'))
#    print(t1.most_similar('i'))

print('')
    
    

'''
viewR = stopWordsFreq['INFJ']
for i in viewR:
    if viewR[i]>3:
        print(i,':',viewR[i])
'''