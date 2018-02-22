#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:10:55 2018

@author: srinidhi
"""

#Neural nets with removal of context specific words and learning rate tweaks
#USING TF-IDF in the feature vectors

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
import random
import datetime
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def isExtroverted(s):
    print(s)
    tempL = ['ESTP','ESTJ','ESFP','ESFJ','ENTP','ENTJ','ENFP','ENFJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(True)
        else:
            ret.append(False)
    return ret

fil = list(csv.reader(open('mbti_big5scores.csv')))
vocabFile = open('top500vocab.txt','r')
tknzr = TweetTokenizer()
#sentAn = SentimentIntensityAnalyzer() #sentiment analyzer
lancaster = LancasterStemmer() #PorterStemmer()
wordnetlem = WordNetLemmatizer()
countVect = CountVectorizer()
saver = tf.train.Saver()

vocab = set()
stopWords = set(stopwords.words('english'))
features = {}
textTrack = {}
puncts = set(string.punctuation)

#Neural network statistics
dispEpoch = 2
saveEveryNEpochs = 5
nEpochs = 10 #should take around a week to run, assuming 10 minutes per epoch
#impossibly high number of epochs, so it will converge and break out instead

nSamples = 3194
nHiddenL1 = 1000
nHiddenL2 = 200
#nHiddenL3 = 20
lr = 0.001
nClasses = 16
testSize = 798
batchSize = 500

rownum = 0

vocab = set(vocabFile.read().strip().split(' '))
nInputs = nWords = len(vocab) #Size of input layer = length of vocabulary

dataTemp = {'E':[], 'I':[]}

startTime = time.time()
print('Starting preprocessing')
for j in fil[1:]:
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
    dataTemp[j[0][0]].append([j[0], ' '.join(temp2)])
endTime = time.time()
print('Preprocessed', str(endTime - startTime))
temp1 = random.sample(dataTemp['I'], 1996)
dataTemp = random.sample(dataTemp['E'] + temp1, 1996 * 2 - 1)
data = np.array(dataTemp)
targetTemp = data[:,0]
target = np.where(isExtroverted(targetTemp), 1, 0)
data = data[:,1]

xTrainCounts = countVect.fit_transform(data) #default: unigram. Can tell it which n-gram you want
#print(countVect.get_feature_names(),len(countVect.get_feature_names()))

tfidfTransformer = TfidfTransformer()
xTrainTfidf = tfidfTransformer.fit_transform(xTrainCounts)

print('Feature vecs done and ready for MLP')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

#X_train, X_test, y_train, y_test = train_test_split(xTrainTfidf, target)
X_train, X_test, y_train, y_test = xTrainTfidf[:nSamples], xTrainTfidf[nSamples:], target[:nSamples], target[nSamples:],

print('Training')

mlp = MLPClassifier(hidden_layer_sizes=(200,50,16),max_iter=500,verbose=True,activation='relu')
print(mlp.fit(X_train, y_train))
predictions = mlp.predict(X_test)
cm = confusion_matrix(y_test,predictions)
print(cm)
print(classification_report(y_test,predictions))
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
print('Accuracy:',accuracy)
















