# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:39:27 2018

@author: Srinidhi
"""

#RandomForestClassifier

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

stopWords = set(stopwords.words('english'))
puncts = set(string.punctuation)
vocab = set(vocabFile.read().strip().split(' '))
nWords = len(vocab)

dataTemp = np.load('16to40withliwc.npy')
liwcTemp = np.load('16to40ofliwc.npy')

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

xTrainTfidf = []
for j in range(len(xTfidf)):
    xTrainTfidf.append(np.concatenate((xTfidf[j], liwcTemp[j])))
xTrainTfidf = np.array(xTrainTfidf)

#xTrainTfidf = np.array(xTfidf)

xTrain, xTest = xTrainTfidf[:nSamples], xTrainTfidf[nSamples:]
yTrainE, yTestE = targetE[:nSamples], targetE[nSamples:]
yTrainN, yTestN = targetN[:nSamples], targetN[nSamples:]
yTrainT, yTestT = targetT[:nSamples], targetT[nSamples:]
yTrainJ, yTestJ = targetJ[:nSamples], targetJ[nSamples:]

clf = RandomForestClassifier()
clf.fit(xTrain, yTrainE)
predE = clf.predict(xTest)

clf = RandomForestClassifier()
clf.fit(xTrain, yTrainN)
predN = clf.predict(xTest)

clf = RandomForestClassifier()
clf.fit(xTrain, yTrainT)
predT = clf.predict(xTest)

clf = RandomForestClassifier()
clf.fit(xTrain, yTrainJ)
predJ = clf.predict(xTest)

accE = accN = accT = accJ = acc = 0
for i in range(len(xTest)):
    c = 0
    if predE[i] == yTestE[i]:
        accE += 1
    if predN[i] == yTestN[i]:
        accN += 1
    if predT[i] == yTestT[i]:
        accT += 1
    if predJ[i] == yTestJ[i]:
        accJ += 1
    if predE[i] == yTestE[i] and predN[i] == yTestN[i] and predT[i] == yTestT[i] and predJ[i] == yTestJ[i]:
        acc += 1

print('Accuracy (E/I):', str(accE / len(xTest)))
print('Accuracy (S/N):', str(accN / len(xTest)))
print('Accuracy (T/F):', str(accT / len(xTest)))
print('Accuracy (J/P):', str(accJ / len(xTest)))
print('Overall accuracy:', str(acc / len(xTest)))

'''
Bigrams with liwc:
Accuracy (E/I): 0.7695488721804511
Accuracy (S/N): 0.8567669172932331
Accuracy (T/F): 0.6413533834586466
Accuracy (J/P): 0.6443609022556391
Overall accuracy: 0.2879699248120301

Bigrams without liwc:
Accuracy (E/I): 0.781578947368421
Accuracy (S/N): 0.85
Accuracy (T/F): 0.6379699248120301
Accuracy (J/P): 0.6349624060150376
Overall accuracy: 0.27030075187969926

Unigrams with liwc:
Accuracy (E/I): 0.7845864661654135
Accuracy (S/N): 0.8639097744360902
Accuracy (T/F): 0.6981203007518797
Accuracy (J/P): 0.643609022556391
Overall accuracy: 0.3206766917293233

Unigrams without liwc:
Accuracy (E/I): 0.7842105263157895
Accuracy (S/N): 0.869172932330827
Accuracy (T/F): 0.6718045112781955
Accuracy (J/P): 0.6597744360902256
Overall accuracy: 0.3112781954887218
'''