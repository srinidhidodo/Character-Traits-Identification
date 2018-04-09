# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:47:43 2018

@author: Srinidhi
"""

#Neural nets with removal of context specific words and learning rate tweaks
#Trying combinations of TF-IDF, LIWC, Emolex and ConceptNet in the feature vectors
#For big 5 dataset (authors)
#Using keras, and softmax output
#Experimenting with Dropout

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
import string
import random
import datetime
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential

def isExtroverted(s):
    #print(s)
    ret = []
    for i in s:
        if i == 'y':
            ret.append(np.array([1,0]))
        else:
            ret.append(np.array([0,1]))
    return ret


#fil = list(csv.reader(open('mbti_big5scores_LIWC.csv', encoding = 'utf8')))
#vocabFile = open('top500vocab_bigrams.txt','r')
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

nSamples = 1728 #For MBTI classes
nHiddenL1 = 1000
nHiddenL2 = 200
#nHiddenL3 = 20
lr = 0.001
#testSize = 5 #For MBTI classes

rownum = 0

#vocab = set(vocabFile.read().strip().split('\n'))
#nWords = len(vocab) #Size of input layer = length of vocabulary

endTime = time.time()
print('Importing and setup time:', str(endTime - startTime))

print('Starting preprocessing')
startTime = time.time()
#fil = fil[1:]
#liwcTemp = [[] for i in fil[0][8:]]
dataTemp = np.load('16to40withliwc_essays.npy')
liwcTemp = np.load('16to40ofliwc_essays.npy') #LIWC components
emolexTemp = np.load('emolex_vector_essays.npy')
conceptTemp = np.load('concept_vector_essays.npy')

startTime = time.time()
dataI = random.sample(range(len(dataTemp)), len(dataTemp))
data = [dataTemp[i] for i in dataI]
liwcTemp = [liwcTemp[i] for i in dataI]
data = np.array(data)
targetTemp = data[:,0]

#Individual neural networks setup
targetO = np.where(isExtroverted(data[:,6]), 1, 0)
targetC = np.where(isExtroverted(data[:,5]), 1, 0)
targetE = np.where(isExtroverted(data[:,2]), 1, 0)
targetA = np.where(isExtroverted(data[:,4]), 1, 0)
targetN = np.where(isExtroverted(data[:,3]), 1, 0)
data = data[:,1]

endTime = time.time()
#print('Sampling time:', str(endTime - startTime))

#print('Feature vectors')
startTime = time.time()
xTrainCounts = countVect.fit_transform(data) #default: unigram. Can tell it which n-gram you want
tfidfTransformer = TfidfTransformer()
xTfidf = tfidfTransformer.fit_transform(xTrainCounts).toarray()
tempLen = len(xTfidf[0])
#print(xTrainTfidf[0])
liwcTemp = np.array(liwcTemp)
#print(liwcTemp[0])

xTrainTfidf = []
for j in range(len(xTfidf)):
    xTrainTfidf.append(np.concatenate((xTfidf[j], liwcTemp[j])))
xTfidf = np.array(xTrainTfidf)

xTrainTfidf = []
for j in range(len(xTfidf)):
    xTrainTfidf.append(np.concatenate((xTfidf[j], emolexTemp[j])))
xTfidf = np.array(xTrainTfidf)

xTrainTfidf = []
for j in range(len(xTfidf)):
    xTrainTfidf.append(np.concatenate((xTfidf[j], conceptTemp[j])))
xTrainTfidf = np.array(xTrainTfidf)
#xTrainTfidf = np.array(xTfidf)
#print(nWords, tempLen, len(xTrainTfidf[0]))

svd = TruncatedSVD(n_components=500, n_iter=5, random_state=42)
xTrainTfidf = svd.fit_transform(xTrainTfidf)

xTrain, xTest = xTrainTfidf[:nSamples], xTrainTfidf[nSamples:]
yTrainO, yTestO = targetO[:nSamples], targetO[nSamples:]
yTrainN, yTestN = targetN[:nSamples], targetN[nSamples:]
yTrainC, yTestC = targetC[:nSamples], targetC[nSamples:]
yTrainE, yTestE = targetE[:nSamples], targetE[nSamples:]
yTrainA, yTestA = targetA[:nSamples], targetA[nSamples:]

mlpE = Sequential()
mlpE.add(Dense(200, activation='relu', input_shape=(len(xTrain[0]),)))
mlpE.add(Dropout(0.5))
mlpE.add(Dense(50, activation='relu'))
mlpE.add(Dropout(0.5))
mlpE.add(Dense(16, activation='relu'))
mlpE.add(Dropout(0.5))
mlpE.add(Dense(units=2, activation='softmax'))
mlpE.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
mlpE.fit(xTrain, yTrainE, epochs=100)

mlpO = Sequential()
mlpO.add(Dense(200, activation='relu', input_shape=(len(xTrain[0]),)))
mlpO.add(Dropout(0.5))
mlpO.add(Dense(50, activation='relu'))
mlpO.add(Dropout(0.5))
mlpO.add(Dense(16, activation='relu'))
mlpO.add(Dropout(0.5))
mlpO.add(Dense(units=2, activation='softmax'))
mlpO.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
mlpO.fit(xTrain, yTrainO, epochs=100)

mlpC = Sequential()
mlpC.add(Dense(200, activation='relu', input_shape=(len(xTrain[0]),)))
mlpC.add(Dropout(0.5))
mlpC.add(Dense(50, activation='relu'))
mlpC.add(Dropout(0.5))
mlpC.add(Dense(16, activation='relu'))
mlpC.add(Dropout(0.5))
mlpC.add(Dense(units=2, activation='softmax'))
mlpC.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
mlpC.fit(xTrain, yTrainC, epochs=100)

mlpA = Sequential()
mlpA.add(Dense(200, activation='relu', input_shape=(len(xTrain[0]),)))
mlpA.add(Dropout(0.5))
mlpA.add(Dense(50, activation='relu'))
mlpA.add(Dropout(0.5))
mlpA.add(Dense(16, activation='relu'))
mlpA.add(Dropout(0.5))
mlpA.add(Dense(units=2, activation='softmax'))
mlpA.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
mlpA.fit(xTrain, yTrainA, epochs=100)

mlpN = Sequential()
mlpN.add(Dense(200, activation='relu', input_shape=(len(xTrain[0]),)))
mlpN.add(Dropout(0.5))
mlpN.add(Dense(50, activation='relu'))
mlpN.add(Dropout(0.5))
mlpN.add(Dense(16, activation='relu'))
mlpN.add(Dropout(0.5))
mlpN.add(Dense(units=2, activation='softmax'))
mlpN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
mlpN.fit(xTrain, yTrainE, epochs=100)


startTime = time.time()
predTempE = mlpE.predict(xTest)
predTempN = mlpN.predict(xTest)
predTempO = mlpO.predict(xTest)
predTempC = mlpC.predict(xTest)
predTempA = mlpA.predict(xTest)

predO = predC = predE = predA = predN = []
for i in range(len(xTest)):
    if predTempE[i][0] >= predTempE[i][1]:
        predE.append([1,0])
    elif predTempE[i][0] < predTempE[i][1]:
        predE.append([0,1])
        
    if predTempO[i][0] >= predTempO[i][1]:
        predO.append([1,0])
    elif predTempO[i][0] < predTempO[i][1]:
        predO.append([0,1])
        
    if predTempC[i][0] >= predTempC[i][1]:
        predC.append([1,0])
    elif predTempC[i][0] < predTempC[i][1]:
        predC.append([0,1])
        
    if predTempA[i][0] >= predTempA[i][1]:
        predA.append([1,0])
    elif predTempA[i][0] < predTempA[i][1]:
        predA.append([0,1])
        
    if predTempN[i][0] >= predTempN[i][1]:
        predN.append([1,0])
    elif predTempN[i][0] < predTempN[i][1]:
        predN.append([0,1])

print('O:',mlpO.evaluate(xTest, yTestO, verbose=0)[1])
print('C;',mlpC.evaluate(xTest, yTestC, verbose=0)[1])
print('E:',mlpE.evaluate(xTest, yTestE, verbose=0)[1])
print('A:',mlpA.evaluate(xTest, yTestA, verbose=0)[1])
print('N:',mlpN.evaluate(xTest, yTestN, verbose=0)[1])

    

'''
Dropout = 0.25
SVD to 300: 51.6%
SVD to 100: 50%
No SVD: 52%

SVD to 500:
O: 0.522972972972973
C; 0.522972972972973
E: 0.504054054054054
A: 0.5581081081081081
N: 0.49324324324324326

Without tfidf:
O: 0.49594594594594593
C; 0.522972972972973
E: 0.5135135135135135
A: 0.5567567567567567
N: 0.4864864864864865

'''