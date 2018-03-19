# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 12:47:30 2018

@author: Srinidhi
"""

#Trying SVM

import time
startTime = time.time()

#import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import string
import random
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def isExtroverted(s):
    #print(s)
    tempL = ['ESTP','ESTJ','ESFP','ESFJ','ENTP','ENTJ','ENFP','ENFJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(1)
        else:
            ret.append(0)
    return ret

def isIntuition(s):
    tempL = ['ENTP','ENTJ','ENFP','ENFJ','INTP','INTJ','INFP','INFJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(1)
        else:
            ret.append(0)
    return ret

def isThinking(s):
    tempL = ['ENTP','ENTJ','ESTP','ESTJ','INTP','INTJ','ISTP','ISTJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(1)
        else:
            ret.append(0)
    return ret

def isJudging(s):
    tempL = ['ENFJ','ENTJ','ESTJ','ESFJ','INFJ','INTJ','ISTJ','ISFJ']
    ret = []
    for i in s:
        if i in tempL:
            ret.append(1)
        else:
            ret.append(0)
    return ret

vocabFile = open('top500vocab_bigrams.txt','r')
tknzr = TweetTokenizer()
#sentAn = SentimentIntensityAnalyzer() #sentiment analyzer
lancaster = LancasterStemmer() #PorterStemmer()
wordnetlem = WordNetLemmatizer()
countVect = CountVectorizer()

stopWords = set(stopwords.words('english'))
puncts = set(string.punctuation)
vocab = set(vocabFile.read().strip().split(' '))
nWords = len(vocab)

dataTemp = np.load('16to40withliwc_bigrams.npy')
liwcTemp = np.load('16to40ofliwc_bigrams.npy')
nRuns = 10
avgAcc = 0

for run in range(nRuns):
    print('\nRun', run)
    dataI = random.sample(range(len(dataTemp)), len(dataTemp))
    data = [dataTemp[i] for i in dataI]
    liwcTemp = [liwcTemp[i] for i in dataI]
    data = np.array(data)
    targetTemp = data[:,0]
    nSamples = 6000
    
    targetE = np.where(isExtroverted(targetTemp), 1, 0)
    targetN = np.where(isIntuition(targetTemp), 1, 0)
    targetT = np.where(isThinking(targetTemp), 1, 0)
    targetJ = np.where(isJudging(targetTemp), 1, 0)
    data = data[:,1]
    xTrainCounts = countVect.fit_transform(data) #default: unigram. Can tell it which n-gram you want
    tfidfTransformer = TfidfTransformer()
    xTfidf = tfidfTransformer.fit_transform(xTrainCounts).toarray()
    tempLen = len(xTfidf[0])
    #print(xTrainTfidf[0])
    liwcTemp = np.array(liwcTemp)
    #print(liwcTemp[0])
    '''
    xTrainTfidf = []
    for j in range(len(xTfidf)):
        xTrainTfidf.append(np.concatenate((xTfidf[j], liwcTemp[j])))
    xTrainTfidf = np.array(xTrainTfidf)
    '''
    xTrainTfidf = np.array(xTfidf)
    
    xTrain, xTest = xTrainTfidf[:nSamples], xTrainTfidf[nSamples:]
    yTrainE, yTestE = targetE[:nSamples], targetE[nSamples:]
    yTrainN, yTestN = targetN[:nSamples], targetN[nSamples:]
    yTrainT, yTestT = targetT[:nSamples], targetT[nSamples:]
    yTrainJ, yTestJ = targetJ[:nSamples], targetJ[nSamples:]
    
    clf = svm.SVC(kernel = 'linear')
    print(clf.fit(xTrain, yTrainE))
    testResE = clf.predict(xTest)
    
    clf = svm.SVC(kernel = 'linear')
    print(clf.fit(xTrain, yTrainN))
    testResN = clf.predict(xTest)
    
    clf = svm.SVC(kernel = 'linear')
    print(clf.fit(xTrain, yTrainT))
    testResT = clf.predict(xTest)
    
    clf = svm.SVC(kernel = 'linear')
    print(clf.fit(xTrain, yTrainJ))
    testResJ = clf.predict(xTest)
    
    accuracy = 0
    for i in range(len(xTest)):
        if testResE[i] == yTestE[i] and testResN[i] == yTestN[i] and testResT[i] == yTestT[i] and testResJ[i] == yTestJ[i]:
            accuracy += 1
    
    accuracy /= len(xTest)
    avgAcc += accuracy
    print('Accuracy:',accuracy)

avgAcc /= nRuns
print('Avg accuracy:',avgAcc)

'''
Accuracy(E/I): 0.8511278195488722
Accuracy(S/N): 0.8511278195488722
Accuracy(overall): 0.5597744360902256

With 16:40 liwc
10 runs: Avg accuracy: 0.5480827067669172

Without liwc
10 runs: Avg accuracy: 0.5563909774436091

Bigrams with 16:40 liwc
Accuracy: 0.4191729323308271

Bigrams without liwc
Accuracy: 0.42142857142857143

'''




