# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:47:43 2018

@author: Srinidhi
"""
#Testing for full MBTI type, not just components of it.
#USING ONLY LIWC

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
nRuns = 10
ETup = (90, 20, 16, 8)
NTup = (90, 20, 16, 8)
TTup = (20, 16, 8, 4)
JTup = (16, 4, 2)

rownum = 0

vocab = set(vocabFile.read().strip().split(' '))
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

for j in fil:
    '''
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
    '''
    #liwcTemp.append([np.float(k) for k in j[8:]])
    dataTemp.append([j[0]]+[np.float(k) for k in j[16:40]])
endTime = time.time()
print('Preprocessing time:', str(endTime - startTime))

avgAcc = 0

for run in range(nRuns):
    print('\nRun',run)
    #print('Sampling for training')
    startTime = time.time()
    data = random.sample(dataTemp, len(dataTemp))
    #data = np.array(data)
    targetTemp = np.array([i[0] for i in data])
    
    print(targetTemp)
    
    #Individual neural networks setup
    targetE = np.where(isExtroverted(targetTemp), 1, 0)
    targetN = np.where(isIntuition(targetTemp), 1, 0)
    targetT = np.where(isThinking(targetTemp), 1, 0)
    targetJ = np.where(isJudging(targetTemp), 1, 0)
    data = np.array([i[1:] for i in data])
    
    endTime = time.time()
    #print('Sampling time:', str(endTime - startTime))
    
    #print('Feature vectors')
    startTime = time.time()
    '''
    xTrainCounts = countVect.fit_transform(data) #default: unigram. Can tell it which n-gram you want
    tfidfTransformer = TfidfTransformer()
    xTfidf = tfidfTransformer.fit_transform(xTrainCounts).toarray()
    '''
    xTrainTfidf = np.array(data)
    
    tempLen = len(xTrainTfidf[0])
    #print(xTrainTfidf[0])
    '''
    liwcTemp = np.array(liwcTemp)
    #print(liwcTemp[0])
    xTrainTfidf = []
    
    for j in range(len(xTfidf)):
        xTrainTfidf.append(np.concatenate((xTfidf[j], liwcTemp[j])))
    xTrainTfidf = np.array(xTrainTfidf)
    '''
    #print(nWords, tempLen, len(xTrainTfidf[0]))
    
    xTrain, xTest = xTrainTfidf[:nSamples], xTrainTfidf[nSamples:]
    yTrainE, yTestE = targetE[:nSamples], targetE[nSamples:]
    yTrainN, yTestN = targetN[:nSamples], targetN[nSamples:]
    yTrainT, yTestT = targetT[:nSamples], targetT[nSamples:]
    yTrainJ, yTestJ = targetJ[:nSamples], targetJ[nSamples:]
    endTime = time.time()
    #print('Feature vector time:', str(endTime - startTime))
    
    #print('Training')
    startTime = time.time()
    
    #print('FOR E/I:')
    mlpE = MLPClassifier(hidden_layer_sizes=ETup,max_iter=500,activation='relu')
    mlpE.fit(xTrain, yTrainE)
    
    #print('FOR S/N:')
    mlpN = MLPClassifier(hidden_layer_sizes=NTup,max_iter=500,activation='relu')
    mlpN.fit(xTrain, yTrainN)
    
    #print('FOR T/F:')
    mlpT = MLPClassifier(hidden_layer_sizes=TTup,max_iter=500,activation='relu')
    mlpT.fit(xTrain, yTrainT)
    
    #print('FOR J/P:')
    mlpJ = MLPClassifier(hidden_layer_sizes=JTup,max_iter=500,activation='relu')
    mlpJ.fit(xTrain, yTrainJ)
    
    endTime = time.time()
    #print('Training time:', str(endTime - startTime))
    
    #print('Testing')
    startTime = time.time()
    predictionsE = mlpE.predict(xTest)
    predictionsN = mlpN.predict(xTest)
    predictionsT = mlpT.predict(xTest)
    predictionsJ = mlpJ.predict(xTest)
    
    cm = confusion_matrix(yTestE,predictionsE)
    #print(cm)
    #print(classification_report(yTestE,predictionsE))
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
    print('Testing Accuracy (E/I):',accuracy)
    
    cm = confusion_matrix(yTestN,predictionsN)
    #print(cm)
    #print(classification_report(yTestN,predictionsN))
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
    print('Testing Accuracy (S/N):',accuracy)
    
    cm = confusion_matrix(yTestT,predictionsT)
    #print(cm)
    #print(classification_report(yTestT,predictionsT))
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
    print('Testing Accuracy (T/F):',accuracy)
    
    cm = confusion_matrix(yTestJ,predictionsJ)
    #print(cm)
    #print(classification_report(yTestJ,predictionsJ))
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
    print('Testing Accuracy (J/P):',accuracy)
    
    correct = 0
    total = 0
    
    for i in range(len(yTestE)):
        if yTestE[i] == predictionsE[i] and yTestN[i] == predictionsN[i] and yTestT[i] == predictionsT[i] and yTestJ[i] == predictionsJ[i]:
            correct += 1
        total += 1
    
    accuracy = correct / total
    avgAcc += accuracy
    
    print('Overall accuracy:',accuracy)

avgAcc /= nRuns
print('E/I:',ETup)
print('S/N:', NTup)
print('T/F:', TTup)
print('J/P:', JTup)
print('Final average accuracy:',avgAcc)

'''
ONLY LIWC:

LIWC [8:]
E/I: (90, 20, 16)
S/N: (90, 20, 16)
T/F: (90, 20, 16)
J/P: (90, 20, 16)
Final average accuracy: 0.2509774436090226

LIWC [8:]
E/I: (90, 20, 16)
S/N: (90, 20, 16)
T/F: (90, 20, 16)
J/P: (40, 16)
Final average accuracy: 0.23033834586466165

LIWC[16:40]
E/I: (90, 20, 16)
S/N: (90, 20, 16)
T/F: (90, 20, 16)
J/P: (40, 16)
Final average accuracy: 0.2351127819548872

[16:40]
E/I: (90, 20, 16)
S/N: (90, 20, 16)
T/F: (90, 20, 16)
J/P: (90, 20, 16)
Final average accuracy: 0.23740601503759398

[16:40]
E/I: (90, 20, 16)
S/N: (90, 20, 16)
T/F: 16
J/P: 16
Final average accuracy: 0.23774436090225565

[16:40]
E/I: (90, 20, 16)
S/N: (90, 20, 16)
T/F: (16, 8)
J/P: (16, 8)
Final average accuracy: 0.24176691729323313

[16:40]
E/I: (90, 20, 16, 8)
S/N: (90, 20, 16, 8)
T/F: (20, 16, 8, 4)
J/P: (16, 4, 2)
Final average accuracy: 0.23169172932330823
'''






