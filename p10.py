#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:36:59 2018

@author: srinidhi
"""

#Removing context specific words; creating vocabulary files

import csv
import re
#import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import string
import time

fil = list(csv.reader(open('mbti_big5scores.csv')))
tknzr = TweetTokenizer()
#sentAn = SentimentIntensityAnalyzer() #sentiment analyzer
lancaster = LancasterStemmer() #PorterStemmer()
wordnetlem = WordNetLemmatizer()
puncts = set(string.punctuation)

counts = {}

startTime = time.time()
for j in fil[1:]:
    i = j[1]
    temp1 = i.split('|||')
    for i in temp1:
        x = re.sub(r'https?\S+', '', re.sub(r'\w*\d\w*', '', i).strip()).strip()
        tok = list(tknzr.tokenize(x))
        for word in tok:
            word = lancaster.stem(wordnetlem.lemmatize(word.lower()))
            word = ''.join(ch for ch in word if ch not in puncts)
            if len(word) > 1 or word == 'i':
                counts[word] = 1 if word not in counts else counts[word] + 1

currTime = time.time()
print(len(counts), str(currTime - startTime))

startTime = time.time()
sortCounts = sorted(counts, key = counts.get, reverse = True)
c1 = c2 = c3 = c4 = True

for i in range(len(sortCounts)):
    if counts[sortCounts[i]] <= 1000 and c1:
        print('1000:',i)
        c1 = False
    if counts[sortCounts[i]] <= 500 and c2:
        print('500:',i)
        c2 = False
    if counts[sortCounts[i]] <= 100 and c3:
        print('100:',i)
        c3 = False
    if counts[sortCounts[i]] <= 50 and c4:
        print('50:',i)
        c4 = False
    if counts[sortCounts[i]] <= 25:
        print('25:',i)
        break

currTime = time.time()

print(counts[sortCounts[1000]],str(currTime - startTime))
'''
fileWrite = open('top50vocab.txt','w')
for i in sortCounts[1:5070]:
    fileWrite.write(i+' ')

fileWrite.close()
'''