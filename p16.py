# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:47:43 2018

@author: Srinidhi
"""
#Testing for full MBTI type, not just components of it.
#Using saved preprocessed data
#Neural nets with removal of context specific words and learning rate tweaks
#USING TF-IDF in the feature vectors

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

#fil = list(csv.reader(open('mbti_big5scores_LIWC.csv', encoding = 'utf8')))
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

dataTemp = np.load('16to40withliwc_bigrams.npy')

endTime = time.time()
print('Importing and setup time:', str(endTime - startTime))

print('Starting preprocessing')
startTime = time.time()
#fil = fil[1:]
#liwcTemp = [[] for i in fil[0][8:]]
liwcTemp = np.load('16to40ofliwc_bigrams.npy')
#print(len(liwcTemp))

nRuns = 15
avgAcc = 0
for run in range(1,nRuns + 1):
    print('\nRun', run)
    #print('Sampling for training')
    startTime = time.time()
    dataI = random.sample(range(len(dataTemp)), len(dataTemp))
    data = [dataTemp[i] for i in dataI]
    liwcTemp = [liwcTemp[i] for i in dataI]
    data = np.array(data)
    targetTemp = data[:,0]
    
    #Individual neural networks setup
    targetE = np.where(isExtroverted(targetTemp), 1, 0)
    targetN = np.where(isIntuition(targetTemp), 1, 0)
    targetT = np.where(isThinking(targetTemp), 1, 0)
    targetJ = np.where(isJudging(targetTemp), 1, 0)
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
    xTrainTfidf = np.array(xTrainTfidf)
    
    #xTrainTfidf = np.array(xTfidf)
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
    mlpE = MLPClassifier(hidden_layer_sizes=(200,50,16),max_iter=500,activation='relu')
    mlpE.fit(xTrain, yTrainE)
    
    #print('FOR S/N:')
    mlpN = MLPClassifier(hidden_layer_sizes=(200,50,16),max_iter=500,activation='relu')
    mlpN.fit(xTrain, yTrainN)
    
    #print('FOR T/F:')
    mlpT = MLPClassifier(hidden_layer_sizes=(200,50,16),max_iter=500,activation='relu')
    mlpT.fit(xTrain, yTrainT)
    
    #print('FOR J/P:')
    mlpJ = MLPClassifier(hidden_layer_sizes=(200,50,16),max_iter=500,activation='relu')
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

print('\nAvg acc:',str(avgAcc/nRuns))

'''
BEFORE LIWC:
Testing Accuracy (E/I): 0.8082706766917294
Testing Accuracy (S/N): 0.8819548872180452
Testing Accuracy (T/F): 0.8142857142857143
Testing Accuracy (J/P): 0.749624060150376
Overall accuracy: 0.4718045112781955

AFTER LIWC (all):
Trial 1:
Testing Accuracy (E/I): 0.7725563909774437
Testing Accuracy (S/N): 0.8552631578947368
Testing Accuracy (T/F): 0.7154135338345865
Testing Accuracy (J/P): 0.6048872180451128
Overall accuracy: 0.27556390977443607

Trial 2:
Testing Accuracy (E/I): 0.7909774436090226
Testing Accuracy (S/N): 0.8545112781954888
Testing Accuracy (T/F): 0.7161654135338346
Testing Accuracy (J/P): 0.6590225563909774
Overall accuracy: 0.3304511278195489

LIWC (41:89):
Testing Accuracy (E/I): 0.8285714285714286
Testing Accuracy (S/N): 0.8857142857142857
Testing Accuracy (T/F): 0.8071428571428572
Testing Accuracy (J/P): 0.73796992481203
Overall accuracy: 0.46466165413533833

LIWC (37:70):
Testing Accuracy (E/I): 0.8109022556390978
Testing Accuracy (S/N): 0.881203007518797
Testing Accuracy (T/F): 0.8157894736842105
Testing Accuracy (J/P): 0.7206766917293234
Overall accuracy: 0.45

LIWC (16:40):
Testing Accuracy (E/I): 0.8199248120300752
Testing Accuracy (S/N): 0.8751879699248121
Testing Accuracy (T/F): 0.8342105263157895
Testing Accuracy (J/P): 0.7635338345864662
Overall accuracy: 0.49849624060150377

LIWC (60:70):
Testing Accuracy (E/I): 0.8086466165413534
Testing Accuracy (S/N): 0.8781954887218045
Testing Accuracy (T/F): 0.8161654135338345
Testing Accuracy (J/P): 0.7364661654135338
Overall accuracy: 0.46278195488721807

LIWC (73:87):
Testing Accuracy (E/I): 0.8082706766917294
Testing Accuracy (S/N): 0.8796992481203008
Testing Accuracy (T/F): 0.8041353383458647
Testing Accuracy (J/P): 0.7345864661654136
Overall accuracy: 0.4575187969924812
'''

'''
LIWC (16:40)
10 runs
Avg acc: 0.49030075187969924

Bigrams with liwc[16:40]:
10 runs: Avg acc: 0.3531578947368421

Bigrams without liwc:
4 runs: Avg acc: 0.35413533834586464

'''






