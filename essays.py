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
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
import math
import statistics

def isYes1(s):
	ret = []
	num = []
	for i in s:
		num.append(float(i))
	num = list(sorted(num))
	m = statistics.mean(num)
	for i in s:
		if float(i) <= m:
			ret.append(0)
		else:
			ret.append(1)
	return ret

def isYes(s):
	ret = []
	for i in s:
		if i == "y":
			ret.append(1)
		else:
			ret.append(0)
	return ret
	
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
svd = TruncatedSVD(n_components=300, n_iter=5, random_state=42)


stopWords = set(stopwords.words('english'))
puncts = set(string.punctuation)
vocab = set(vocabFile.read().strip().split(' '))
nWords = len(vocab)

dataTemp = np.load('16to40withliwc_essays.npy')
#dataTemp = np.load('16to40withliwc_twitter_big5.npy')
#print(dataTemp.shape)
liwcTemp = np.load('16to40ofliwc_essays.npy')
#liwcTemp = liwcTemp[:,16:40]
emolexTemp = np.load('emolex_vector_essays.npy')
conceptTemp = np.load('concept_vector_essays.npy')
nRuns = 1
avgAcc = 0


for run in range(nRuns):
    print('\nRun', run)
    dataI = random.sample(range(len(dataTemp)), len(dataTemp))
    data = [dataTemp[i] for i in dataI]
    liwcTemp = [liwcTemp[i] for i in dataI]
    data = np.array(data)
    targetTemp1 = data[:,2]
    targetTemp2 = data[:,3]
    targetTemp3 = data[:,4]
    targetTemp4 = data[:,5]
    targetTemp5 = data[:,6]
    nSamples = 1700
    #nSamples = 6000
    
    targetE = np.where(isYes(targetTemp1), 1, 0)
    targetN = np.where(isYes(targetTemp2), 1, 0)
    targetA = np.where(isYes(targetTemp3), 1, 0)
    targetC = np.where(isYes(targetTemp4), 1, 0)
    targetO = np.where(isYes(targetTemp5), 1, 0)
    
    data = data[:,1]
    import os,shutil
    root="text_files"
    os.chdir(root)
    count = 0
    #print(xTest)
    temp_data = data
    
   
    #print(liwc_predict)
    
    	
    
    			
    #print(count)
    #temp_data = np.array(temp_data)
    xTrainCounts = countVect.fit_transform(data) #default: unigram. Can tell it which n-gram you want
    tfidfTransformer = TfidfTransformer()
    xTfidf = tfidfTransformer.fit_transform(xTrainCounts).toarray()
    #xTfidf = np.load('16to40ofliwc.npy')
    #print(pcaRunner.fit(xTfidf))
    #xTfidf = pcaRunner.transform(xTfidf)
    
    #tempLen = len(xTfidf[0])
    #print(xTrainTfidf[0])
    #liwcTemp = np.array(liwcTemp)
    #print(liwcTemp[0])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    xTrainTfidf = []
    for j in range(len(xTfidf)):
        xTrainTfidf.append(np.concatenate((liwcTemp[j], emolexTemp[j], conceptTemp[j])))
    test_feature = []
    #for j in range(100):
    	#liwcTransform = np.reshape(liwc_predict[j], [1, len(liwc_predict[j])])
    	#print(liwcTransform.shape, emolex_predict[j].shape, concept_predict[j].shape, concept_predict[j][0].shape)
    	#test_feature.append(np.concatenate((liwcTransform[0],emolex_predict[j][0],concept_predict[j][0])))
    #xTrainTfidf = conceptTemp
    #xTrainTfidf = np.array(np.absolute(xTrainTfidf))
    #print(pcaRunner.fit(xTrainTfidf))
    #xTrainTfidf = pcaRunner.transform(xTrainTfidf)
    print(svd.fit(xTrainTfidf))
    xTrainTfidf = svd.transform(xTrainTfidf)
    #test_feature = svd.transform(test_feature)
    xTrainTfidf = np.array(np.absolute(xTrainTfidf))
    #test_feature = np.array(np.absolute(test_feature))
    #xTrainTfidf = np.array(np.absolute(xTfidf))
    
    xTrain, xTest = xTrainTfidf[:nSamples], xTrainTfidf[nSamples:]
    yTrainE, yTestE = targetE[:nSamples], targetE[nSamples:]
    yTrainN, yTestN = targetN[:nSamples], targetN[nSamples:]
    yTrainA, yTestA = targetA[:nSamples], targetA[nSamples:]
    yTrainC, yTestC = targetC[:nSamples], targetC[nSamples:]
    yTrainO, yTestO = targetO[:nSamples], targetO[nSamples:]
    
    
    '''scoreTestE = targetTemp1[nSamples:]
    scoreTestN = targetTemp2[nSamples:]     
    scoreTestA = targetTemp3[nSamples:]
    scoreTestC = targetTemp4[nSamples:]
    scoreTestO = targetTemp5[nSamples:]'''
    
    clf = svm.SVC(kernel = 'linear', probability=True)
    #clf = MultinomialNB()
    print(clf.fit(xTrain, yTrainE))
    testResE = clf.predict(xTest)
    testResEP = clf.predict_proba(xTest)
    trainResEP = clf.predict_proba(xTrain)
    testResEP = testResEP.tolist()
    '''for i in range(100):
    	res = clf.predict_proba(test_feature[i])
    	new_file = (file_name[i].split(".txt"))[0]+"_bigfiveE.npy"
    	np.save(new_file,res)'''
    #for val in testResEP:
    	#print(val)
    
    clf = svm.SVC(kernel = 'linear', probability=True)
    #clf = MultinomialNB()
    print(clf.fit(xTrain, yTrainN))
    testResN = clf.predict(xTest)
    testResNP = clf.predict_proba(xTest)
    trainResNP = clf.predict_proba(xTrain)
    '''for i in range(100):
    	res = clf.predict_proba(test_feature[i])
    	new_file = (file_name[i].split(".txt"))[0]+"_bigfiveN.npy"
    	np.save(new_file,res)'''
    #print(testResNP)
    #for val in testResNP:
    	#print(val)
    
    clf = svm.SVC(kernel = 'linear', probability=True)
    #clf = MultinomialNB()
    print(clf.fit(xTrain, yTrainA))
    testResA = clf.predict(xTest)
    testResAP = clf.predict_proba(xTest)
    trainResAP = clf.predict_proba(xTrain)
    '''for i in range(100):
    	res = clf.predict_proba(test_feature[i])
    	new_file = (file_name[i].split(".txt"))[0]+"_bigfiveA.npy"
    	np.save(new_file,res)'''
    #print(testResAP)
    #for val in testResAP:
    	#print(val)
    
    clf = svm.SVC(kernel = 'linear', probability=True)
    #clf = MultinomialNB()
    print(clf.fit(xTrain, yTrainC))
    testResC = clf.predict(xTest)
    testResCP = clf.predict_proba(xTest)
    trainResCP = clf.predict_proba(xTrain)
    '''for i in range(100):
    	res = clf.predict_proba(test_feature[i])
    	new_file = (file_name[i].split(".txt"))[0]+"_bigfiveC.npy"
    	np.save(new_file,res)'''
    #print(testResCP)
    #for val in testResCP:
    	#print(val)
    
    clf = svm.SVC(kernel = 'linear', probability=True)
    #clf = MultinomialNB()
    print(clf.fit(xTrain, yTrainO))
    testResO = clf.predict(xTest)
    testResOP = clf.predict_proba(xTest)
    trainResOP = clf.predict_proba(xTrain)
    '''for i in range(100):
    	res = clf.predict_proba(test_feature[i])
    	new_file = (file_name[i].split(".txt"))[0]+"_bigfiveO.npy"
    	np.save(new_file,res)'''
    #print(testResOP)
    #for val in testResOP:
    	#print(val)'''
    file_name = []
    for f in os.listdir("."):
    	text = open(f).read()
    	for s in range(len(temp_data)):
    		if temp_data[s] == text:
				if s < len(xTrain):
					

    
    
    
    '''accuracy = 0
    accE = accN = accA = accC = accO = acc = 0

    for i in range(len(xTest)):
        if testResE[i] == yTestE[i]:
            accE += 1
        if testResN[i] == yTestN[i]:
            accN += 1
        if testResA[i] == yTestA[i]:
            accA += 1
        if testResC[i] == yTestC[i]:
            accC += 1
        if testResO[i] == yTestO[i]:
            accO += 1
            
            
      
        
        #if testResE[i] == yTestE[i] and testResN[i] == yTestN[i] and testResT[i] == yTestT[i] and testResJ[i] == yTestJ[i]:
            #accuracy += 1
    
    #accuracy /= len(xTest)
    
    #avgAcc += accuracy
    print('Accuracy E:', str(accE / len(xTest)))
    print('Accuracy N:', str(accN / len(xTest)))
    print('Accuracy A:', str(accA / len(xTest)))
    print('Accuracy C:', str(accC / len(xTest)))
    print('Accuracy O:', str(accO / len(xTest)))
    
    accuracy = 0
    accE = accN = accA = accC = accO = acc = 0.0'''

    '''for i in range(len(xTest)):
    	print("E : ",abs(float(scoreTestE[i])-testResEP[i][0]))
    	print("N : ",abs(float(scoreTestN[i])-testResNP[i][0]))
    	print("A : ",abs(float(scoreTestA[i])-testResAP[i][0]))
    	print("C : ",abs(float(scoreTestC[i])-testResCP[i][0]))
    	print("O : ",abs(float(scoreTestO[i])-testResOP[i][0]))
    	accE += math.pow(abs(float(scoreTestE[i])-testResEP[i][0]),2)
    	accN += math.pow(abs(float(scoreTestN[i])-testResNP[i][0]),2)
    	accA += math.pow(abs(float(scoreTestA[i])-testResAP[i][0]),2)
    	accC += math.pow(abs(float(scoreTestC[i])-testResCP[i][0]),2)
    	accO += math.pow(abs(float(scoreTestO[i])-testResOP[i][0]),2)
            
            
      
        
        #if testResE[i] == yTestE[i] and testResN[i] == yTestN[i] and testResT[i] == yTestT[i] and testResJ[i] == yTestJ[i]:
            #accuracy += 1
    
    #accuracy /= len(xTest)
    
    #avgAcc += accuracy
    print('Accuracy E:', str(math.pow(accE / (len(xTest)),0.5)))
    print('Accuracy N:', str(math.pow(accN / (len(xTest)),0.5)))
    print('Accuracy A:', str(math.pow(accA / (len(xTest)),0.5)))
    print('Accuracy C:', str(math.pow(accC / (len(xTest)),0.5)))
    print('Accuracy O:', str(math.pow(accO / (len(xTest)),0.5)))
    #print('Accuracy:',accuracy)

#avgAcc /= nRuns
#print('Avg accuracy:',avgAcc)'''

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

#1975

'''
tfidf, liwc, emolex
PCA(copy=True, iterated_power='auto', n_components=50, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
Accuracy E: 0.4739583333333333
Accuracy N: 0.5052083333333334
Accuracy A: 0.48828125
Accuracy C: 0.5208333333333334
Accuracy O: 0.5091145833333334

Run 0
PCA(copy=True, iterated_power='auto', n_components=50, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Accuracy E: 0.50390625
Accuracy N: 0.4973958333333333
Accuracy A: 0.546875
Accuracy C: 0.5208333333333334
Accuracy O: 0.5091145833333334


liwc+emolex
Run 0
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
Accuracy E: 0.5442708333333334
Accuracy N: 0.5716145833333334
Accuracy A: 0.5677083333333334
Accuracy C: 0.5403645833333334
Accuracy O: 0.546875

Run 0
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Accuracy E: 0.578125
Accuracy N: 0.5572916666666666
Accuracy A: 0.5533854166666666
Accuracy C: 0.5794270833333334
Accuracy O: 0.57421875

liwc+emolex+conceptnet

Run 0
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
Accuracy E: 0.5286458333333334
Accuracy N: 0.5234375
Accuracy A: 0.5091145833333334
Accuracy C: 0.54296875
Accuracy O: 0.5208333333333334

Run 0
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Accuracy E: 0.5690104166666666
Accuracy N: 0.5364583333333334
Accuracy A: 0.5703125
Accuracy C: 0.5950520833333334
Accuracy O: 0.6263020833333334

Run 0
TruncatedSVD(algorithm='randomized', n_components=120, n_iter=5,
       random_state=42, tol=0.0)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Accuracy E: 0.5885416666666666
Accuracy N: 0.5494791666666666
Accuracy A: 0.5546875
Accuracy C: 0.5872395833333334
Accuracy O: 0.6067708333333334

Run 0
TruncatedSVD(algorithm='randomized', n_components=150, n_iter=5,
       random_state=42, tol=0.0)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Accuracy E: 0.58984375
Accuracy N: 0.5364583333333334
Accuracy A: 0.5403645833333334
Accuracy C: 0.55859375
Accuracy O: 0.6197916666666666

Run 0
TruncatedSVD(algorithm='randomized', n_components=300, n_iter=5,
       random_state=42, tol=0.0)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Accuracy E: 0.5885416666666666
Accuracy N: 0.5794270833333334
Accuracy A: 0.5677083333333334
Accuracy C: 0.5729166666666666
Accuracy O: 0.6276041666666666



Watson

Accuracy E: 0.8451127819548873
Accuracy N: 0.8710526315789474
Accuracy A: 0.8943609022556391
Accuracy C: 0.8575187969924812
Accuracy O: 0.8793233082706767
Accuracy E: 0.4084591698476908
Accuracy N: 0.517285121783892
Accuracy A: 0.4999328458183898
Accuracy C: 0.4207438754694113
Accuracy O: 0.4463863450630666



'''




