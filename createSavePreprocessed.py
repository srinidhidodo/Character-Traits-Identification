# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:05:12 2018

@author: Srinidhi
"""

import time
startTime = time.time()

import csv
import re
#import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import bigrams
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def isExtroverted(s):
    #print(s)
    tempL = ['ESTP','ESTJ','ESFP','ESFJ','ENTP','ENTJ','ENFP','ENFJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(True)
        else:
            ret.append(False)
    return ret

def isIntuition(s):
    tempL = ['ENTP','ENTJ','ENFP','ENFJ','INTP','INTJ','INFP','INFJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(True)
        else:
            ret.append(False)
    return ret

def isThinking(s):
    tempL = ['ENTP','ENTJ','ESTP','ESTJ','INTP','INTJ','ISTP','ISTJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(True)
        else:
            ret.append(False)
    return ret

def isJudging(s):
    tempL = ['ENFJ','ENTJ','ESTJ','ESFJ','INFJ','INTJ','ISTJ','ISFJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(True)
        else:
            ret.append(False)
    return ret

fil = list(csv.reader(open('mbti_big5scores_LIWC.csv', encoding = 'utf8')))
vocabFile = open('top500vocab_bigrams.txt','r')
tknzr = TweetTokenizer()
#sentAn = SentimentIntensityAnalyzer() #sentiment analyzer
lancaster = LancasterStemmer() #PorterStemmer()
wordnetlem = WordNetLemmatizer()
countVect = CountVectorizer()

vocab = set()
stopWords = set(stopwords.words('english'))
features = {}
textTrack = {}
puncts = set(string.punctuation)

nSamples = 6000 #For MBTI classes
nHiddenL1 = 1000
nHiddenL2 = 200
#nHiddenL3 = 20
lr = 0.001
#testSize = 5 #For MBTI classes

rownum = 0

vocab = set(vocabFile.read().strip().split('\n'))
nWords = len(vocab) #Size of input layer = length of vocabulary

dataTemp = []

endTime = time.time()
print('Importing and setup time:', str(endTime - startTime))

print('Starting preprocessing')
startTime = time.time()
fil = fil[1:]
#liwcTemp = [[] for i in fil[0][8:]]
liwcTemp = []
#print(len(liwcTemp))

'''
for j in fil:
    i = j[1]
    temp1 = i.split('|||')
    temp2 = []
    for i in temp1:
        #Removing URLs and numbers
        x = re.sub(r'https?\S+', '', re.sub(r'\w*\d\w*', '', i).strip()).strip()
        #Tokenizing
        tok = list(tknzr.tokenize(x))
        for x in tok:
            tempWord = lancaster.stem(wordnetlem.lemmatize(x.lower()))
            tempWord = ''.join(ch for ch in tempWord if ch not in puncts)
            if len(tempWord) > 1 and tempWord in vocab:
                temp2.append(tempWord)
    dataToAdd = [j[0], ' '.join(temp2)]
    dataTemp.append(dataToAdd)
#    for i in range(len(j[8:])):
#        liwcTemp[i].append(j[i+8])
    liwcTemp.append([np.float(k) for k in j[16:40]])
endTime = time.time()
print('Preprocessing time:', str(endTime - startTime))
'''

for j in fil:
    i = j[1]
    temp1 = i.split('|||')
    temp2 = []
    for i in temp1:
        #Removing URLs and numbers
        x = re.sub(r'https?\S+', '', re.sub(r'\w*\d\w*', 'thisisalink', i).strip()).strip()
        #Tokenizing
        tok = list(tknzr.tokenize(x))
        for x in tok:
            tempWord = lancaster.stem(wordnetlem.lemmatize(x.lower()))
            tempWord = ''.join(ch for ch in tempWord if ch not in puncts)
            if len(tempWord) > 1:
                temp2.append(tempWord)
    #dataToAdd = [j[0], ' '.join(temp2)]
    temp2 = bigrams(temp2)
    dataToAdd = []
    for i in temp2:
        k = ' '.join(i)
        if k in vocab:
            dataToAdd.append(''.join(i))
    dataTemp.append([j[0],' '.join(dataToAdd)])
#    for i in range(len(j[8:])):
#        liwcTemp[i].append(j[i+8])
    liwcTemp.append([np.float(k) for k in j[16:40]])
endTime = time.time()
print('Preprocessing time:', str(endTime - startTime))

np.save('16to40withliwc_bigrams.npy', dataTemp)
np.save('16to40ofliwc_bigrams.npy', liwcTemp)




