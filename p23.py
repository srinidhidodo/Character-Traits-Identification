# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:47:43 2018

@author: Srinidhi
"""
#Neural nets with removal of context specific words and learning rate tweaks
#Trying combinations of TF-IDF, LIWC, Emolex and ConceptNet in the feature vectors
#For MBTI dataset (twitter)
#Using keras, and softmax output
#Experimenting with Dropout

import time
startTime = time.time()

import csv
import re
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
    tempL = ['ESTP','ESTJ','ESFP','ESFJ','ENTP','ENTJ','ENFP','ENFJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(np.array([1,0]))
        else:
            ret.append(np.array([0,1]))
    return np.array(ret)

def isIntuition(s):
    tempL = ['ENTP','ENTJ','ENFP','ENFJ','INTP','INTJ','INFP','INFJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(np.array([1,0]))
        else:
            ret.append(np.array([0,1]))
    return np.array(ret)

def isThinking(s):
    tempL = ['ENTP','ENTJ','ESTP','ESTJ','INTP','INTJ','ISTP','ISTJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(np.array([1,0]))
        else:
            ret.append(np.array([0,1]))
    return np.array(ret)

def isJudging(s):
    tempL = ['ENFJ','ENTJ','ESTJ','ESFJ','INFJ','INTJ','ISTJ','ISFJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(np.array([1,0]))
        else:
            ret.append(np.array([0,1]))
    return np.array(ret)

tknzr = TweetTokenizer()
lancaster = LancasterStemmer() #PorterStemmer()
wordnetlem = WordNetLemmatizer()
countVect = CountVectorizer()

vocab = set()
stopWords = set(stopwords.words('english'))
puncts = set(string.punctuation)

nSamples = 6000 #For MBTI classes

endTime = time.time()
startTime = time.time()
dataTemp = np.load('16to40withliwc.npy')
liwcTemp = np.load('16to40ofliwc.npy') #LIWC components
emolexTemp = np.load('emolex_vector.npy')
conceptTemp = np.load('concept_vector.npy')

startTime = time.time()
dataI = random.sample(range(len(dataTemp)), len(dataTemp))
data = [dataTemp[i] for i in dataI]
liwcTemp = [liwcTemp[i] for i in dataI]
data = np.array(data)
targetTemp = data[:,0]

#Individual neural networks setup
targetE = isExtroverted(data[:,0])
targetN = isIntuition(data[:,0])
targetT = isThinking(data[:,0])
targetJ = isJudging(data[:,0])
data = data[:,1]

endTime = time.time()

startTime = time.time()
xTrainCounts = countVect.fit_transform(data) #default: unigram. Can tell it which n-gram you want
tfidfTransformer = TfidfTransformer()
xTfidf = tfidfTransformer.fit_transform(xTrainCounts).toarray()
tempLen = len(xTfidf[0])
liwcTemp = np.array(liwcTemp)

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

#svd = TruncatedSVD(n_components=300, n_iter=5, random_state=42)
#xTrainTfidf = svd.fit_transform(xTrainTfidf)

xTrain, xTest = xTrainTfidf[:nSamples], xTrainTfidf[nSamples:]
yTrainT, yTestT = targetT[:nSamples], targetT[nSamples:]
yTrainN, yTestN = targetN[:nSamples], targetN[nSamples:]
yTrainJ, yTestJ = targetJ[:nSamples], targetJ[nSamples:]
yTrainE, yTestE = targetE[:nSamples], targetE[nSamples:]

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
mlpO.fit(xTrain, yTrainT, epochs=100)

mlpC = Sequential()
mlpC.add(Dense(200, activation='relu', input_shape=(len(xTrain[0]),)))
mlpC.add(Dropout(0.5))
mlpC.add(Dense(50, activation='relu'))
mlpC.add(Dropout(0.5))
mlpC.add(Dense(16, activation='relu'))
mlpC.add(Dropout(0.5))
mlpC.add(Dense(units=2, activation='softmax'))
mlpC.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
mlpC.fit(xTrain, yTrainJ, epochs=100)

mlpN = Sequential()
mlpN.add(Dense(200, activation='relu', input_shape=(len(xTrain[0]),)))
mlpN.add(Dropout(0.5))
mlpN.add(Dense(50, activation='relu'))
mlpN.add(Dropout(0.5))
mlpN.add(Dense(16, activation='relu'))
mlpN.add(Dropout(0.5))
mlpN.add(Dense(units=2, activation='softmax'))
mlpN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
mlpN.fit(xTrain, yTrainN, epochs=100)


startTime = time.time()
predTempE = mlpE.predict(xTest)
predTempN = mlpN.predict(xTest)
predTempO = mlpO.predict(xTest)
predTempC = mlpC.predict(xTest)

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
        
    if predTempN[i][0] >= predTempN[i][1]:
        predN.append([1,0])
    elif predTempN[i][0] < predTempN[i][1]:
        predN.append([0,1])

print('E/I:',mlpE.evaluate(xTest, yTestE, verbose=0)[1])
print('S/N:',mlpN.evaluate(xTest, yTestN, verbose=0)[1])
print('T/F:',mlpO.evaluate(xTest, yTestT, verbose=0)[1])
print('J/P;',mlpC.evaluate(xTest, yTestJ, verbose=0)[1])
