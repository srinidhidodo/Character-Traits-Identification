#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:28:47 2018

@author: srinidhi
"""

#Trying MLP with LIWC, Emolex and PCA

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import string
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

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

vocabFile = open('top500vocab.txt','r')
tknzr = TweetTokenizer()
#sentAn = SentimentIntensityAnalyzer() #sentiment analyzer
lancaster = LancasterStemmer() #PorterStemmer()
wordnetlem = WordNetLemmatizer()
countVect = CountVectorizer()
pcaRunner = PCA(n_components = 100, svd_solver = 'auto')

stopWords = set(stopwords.words('english'))
puncts = set(string.punctuation)
vocab = set(vocabFile.read().strip().split(' '))
nWords = len(vocab)

dataTemp = np.load('16to40withliwc.npy')
liwcTemp = np.load('16to40ofliwc.npy')
emolexTemp = np.load('emolex_vector.npy')

data = np.array(dataTemp)
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

xTrainTfidf = []
for j in range(len(xTfidf)):
    xTrainTfidf.append(np.concatenate((xTfidf[j], liwcTemp[j])))
xTrainTfidf = np.array(xTrainTfidf)

#xTrainTfidf = np.array(xTfidf)
'''
xTfidf = xTrainTfidf
xTrainTfidf = []
for j in range(len(xTfidf)):
    xTrainTfidf.append(np.concatenate((xTfidf[j], emolexTemp[j])))
xTrainTfidf = np.array(xTrainTfidf)
#xTrainTfidf = np.array(xTfidf)
'''
#print(pcaRunner.fit(xTrainTfidf))
#xTrainTfidf = pcaRunner.transform(xTrainTfidf)

nRuns = 5
avgAcc = 0
for run in range(1,nRuns + 1):
    print('\nRun', run)
    
    dataI = random.sample(range(len(xTrainTfidf)), len(xTrainTfidf))
    #data = [dataTemp[i] for i in dataI]
    xTrainTfidf = [xTrainTfidf[i] for i in dataI]
    targetE = [targetE[i] for i in dataI]
    targetN = [targetN[i] for i in dataI]
    targetT = [targetT[i] for i in dataI]
    targetJ = [targetJ[i] for i in dataI]    
    
    xTrain, xTest = xTrainTfidf[:nSamples], xTrainTfidf[nSamples:]
    yTrainE, yTestE = targetE[:nSamples], targetE[nSamples:]
    yTrainN, yTestN = targetN[:nSamples], targetN[nSamples:]
    yTrainT, yTestT = targetT[:nSamples], targetT[nSamples:]
    yTrainJ, yTestJ = targetJ[:nSamples], targetJ[nSamples:]
    
    xTrain = np.array(xTrain)
    
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
All liwc:
Testing Accuracy (E/I): 0.781203007518797
Testing Accuracy (S/N): 0.8883458646616541
Testing Accuracy (T/F): 0.7353383458646616
Testing Accuracy (J/P): 0.7086466165413534
Overall accuracy: 0.38609022556390976

No liwc:
Testing Accuracy (E/I): 0.8383458646616542
Testing Accuracy (S/N): 0.8992481203007519
Testing Accuracy (T/F): 0.812781954887218
Testing Accuracy (J/P): 0.7699248120300752
Overall accuracy: 0.5157894736842106

With liwc (16:40):
Testing Accuracy (E/I): 0.8533834586466166
Testing Accuracy (S/N): 0.9045112781954887
Testing Accuracy (T/F): 0.8575187969924812
Testing Accuracy (J/P): 0.749624060150376
Overall accuracy: 0.5526315789473685

Avg acc: 0.5545112781954887

'''