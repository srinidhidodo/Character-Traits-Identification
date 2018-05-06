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
vocabFile = open('top500vocab.txt','r')
tknzr = TweetTokenizer()
lancaster = LancasterStemmer() #PorterStemmer()
wordnetlem = WordNetLemmatizer()
countVect = CountVectorizer()

vocab = set()
stopWords = set(stopwords.words('english'))
puncts = set(string.punctuation)

vocab = set(vocabFile.read().strip().split(' '))
nWords = len(vocab) #Size of input layer = length of vocabulary
dataTemp = []

fil = fil[1:]
liwcTemp = []

for j in fil:
    i = temp1 = j[1]
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
    dataToAdd = [j[0], ' '.join(temp2)] + j[2:7]
    dataTemp.append(dataToAdd)
    liwcTemp.append([np.float(k) for k in j[8:]])


np.save('16to40withliwc.npy', dataTemp)
np.save('16to40ofliwc.npy', liwcTemp)
