# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 12:47:30 2018

@author: Srinidhi
"""

#Trying SVM with LIWC, Emolex, PCA

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
from sklearn.decomposition import PCA

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

xTfidf = liwcTemp
xTrainTfidf = []
for j in range(len(xTfidf)):
    xTrainTfidf.append(np.concatenate((xTfidf[j], emolexTemp[j])))
xTrainTfidf = np.array(xTrainTfidf)
#xTrainTfidf = np.array(xTfidf)

print(pcaRunner.fit(xTrainTfidf))
xTrainTfidf = pcaRunner.transform(xTrainTfidf)

nRuns = 2
avgAcc = 0

for run in range(nRuns):
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
    accE = accN = accT = accJ = acc = 0

    for i in range(len(xTest)):
        if testResE[i] == yTestE[i]:
            accE += 1
        if testResN[i] == yTestN[i]:
            accN += 1
        if testResT[i] == yTestT[i]:
            accT += 1
        if testResJ[i] == yTestJ[i]:
            accJ += 1        
        if testResE[i] == yTestE[i] and testResN[i] == yTestN[i] and testResT[i] == yTestT[i] and testResJ[i] == yTestJ[i]:
            accuracy += 1
    
    accuracy /= len(xTest)
    avgAcc += accuracy
    print('Accuracy (E/I):', str(accE / len(xTest)))
    print('Accuracy (S/N):', str(accN / len(xTest)))
    print('Accuracy (T/F):', str(accT / len(xTest)))
    print('Accuracy (J/P):', str(accJ / len(xTest)))
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
Accuracy (E/I): 0.8533834586466166
Accuracy (S/N): 0.8845864661654136
Accuracy (T/F): 0.8593984962406015
Accuracy (J/P): 0.7906015037593985
Accuracy: 0.556390977443609

Bigrams with 16:40 liwc
Accuracy (E/I): 0.8161654135338345
Accuracy (S/N): 0.8699248120300752
Accuracy (T/F): 0.7759398496240602
Accuracy (J/P): 0.7052631578947368
Accuracy: 0.42894736842105263

Bigrams without liwc
Accuracy (E/I): 0.8090225563909774
Accuracy (S/N): 0.8804511278195488
Accuracy (T/F): 0.7676691729323308
Accuracy (J/P): 0.7018796992481203
Accuracy: 0.42105263157894735
'''

'''
PCA 1500 to 100
Unigrams, with liwc [16:40]:
Accuracy (E/I): 0.8390977443609022
Accuracy (S/N): 0.8796992481203008
Accuracy (T/F): 0.8620300751879699
Accuracy (J/P): 0.7800751879699248
Accuracy: 0.5409774436090226

Unigrams with all liwc:
Accuracy (E/I): 0.8357142857142857
Accuracy (S/N): 0.8789473684210526
Accuracy (T/F): 0.8522556390977444
Accuracy (J/P): 0.7766917293233083
Accuracy: 0.5236842105263158

Unigrams without liwc:
Accuracy (E/I): 0.8357142857142857
Accuracy (S/N): 0.869172932330827
Accuracy (T/F): 0.8556390977443609
Accuracy (J/P): 0.7898496240601504
Accuracy: 0.5360902255639097

Bigrams, with liwc[16:40]:
Avg accuracy: 0.4351503759398496
'''




