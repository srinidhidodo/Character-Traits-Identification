#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:10:55 2018

@author: srinidhi
"""

#Neural nets with removal of context specific words and learning rate tweaks

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
#from sklearn.feature_extraction.text import CountVectorizer

def index(theList, theItem):
    if theItem in theList:
        return theList.index(theItem)
    return -1

def indexOfMbtiType(theType):
    indexList = ['ESTP','ESTJ','ESFP','ESFJ','ENTP','ENTJ','ENFP','ENFJ','ISTP','ISTJ','ISFP','ISFJ','INTP','INTJ','INFP','INFJ']
    return indexList.index(theType)

def buildVocab(tokens): #vocab is a set, tokens is a list
    tempList = []
    for i in tokens:
        #if i not in stopWords:
        tempWord = lancaster.stem(wordnetlem.lemmatize(i.lower()))
        tempWord = ''.join(ch for ch in tempWord if ch not in puncts)
        if len(tempWord) > 1:
            tempList.append(tempWord)
    return tempList

def buildFeatureVector(tokens):
    temp = list(vocab)
    fVec = [0 for i in temp]
    #print(len(fVec))
    for i in tokens:
        if i in temp:
            fVec[index(temp, i)] += 1
    return fVec

fil = list(csv.reader(open('mbti_big5scores.csv')))
vocabFile = open('top500vocab.txt','r')
tknzr = TweetTokenizer()
#sentAn = SentimentIntensityAnalyzer() #sentiment analyzer
lancaster = LancasterStemmer() #PorterStemmer()
wordnetlem = WordNetLemmatizer()
#vectorizer = CountVectorizer()

vocab = set()
stopWords = set(stopwords.words('english'))
features = {}
textTrack = {}
puncts = set(string.punctuation)

#Neural network statistics
dispEpoch = 2
nEpochs = 25
nSamples = 7100
nHiddenL1 = 1000
nHiddenL2 = 200
#nHiddenL3 = 20
lr = 0.001
nClasses = 16
testSize = 1500
batchSize = 500

rownum = 0

vocab = set(vocabFile.read().strip().split(' '))
data = random.sample(fil[1:], nSamples + testSize)

def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1_activation = tf.nn.sigmoid(layer_1_addition)
    
    #Hidden layer with ReLU
    layer_2_multiplication = tf.matmul(layer_1_activation, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2_activation = tf.nn.sigmoid(layer_2_addition)
    
    '''
    layer_3_multiplication = tf.matmul(layer_2_activation, weights['h3'])
    layer_3_addition = tf.add(layer_3_multiplication, biases['b3'])
    layer_3_activation = tf.nn.relu(layer_3_addition)
    '''
    # Output layer with linear activation
    out_layer_multiplication = tf.matmul(layer_2_activation, weights['out']) #to L2
    out_layer_addition = out_layer_multiplication + biases['out']
    
    return out_layer_addition


#Size of input layer = length of vocabulary
nInputs = nWords = len(vocab)

weights = {
    'h1': tf.Variable(tf.random_normal([nInputs, nHiddenL1])),
    'h2': tf.Variable(tf.random_normal([nHiddenL1, nHiddenL2])),
    #'h3': tf.Variable(tf.random_normal([nHiddenL2, nHiddenL3])),
    'out': tf.Variable(tf.random_normal([nHiddenL2, nClasses]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([nHiddenL1])),
    'b2': tf.Variable(tf.random_normal([nHiddenL2])),
    #'b3': tf.Variable(tf.random_normal([nHiddenL3])),
    'out': tf.Variable(tf.random_normal([nClasses]))
}
ipTensor = tf.placeholder(tf.float32, [None, nInputs], name = 'input')
opTensor = tf.placeholder(tf.float32, [None, nClasses], name = 'output')
prediction = multilayer_perceptron(ipTensor, weights, biases)
entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=opTensor)
cost = tf.reduce_mean(entropy_loss) #cost function same as loss function
opt = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)

startTime = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, nEpochs + 1):
        avgCost = 0
        fVec = []
        opVec = []
        for j in data[1:nSamples]:
            i = j[1]
            rownum += 1
            temp1 = i.split('|||')
            if j[0] not in textTrack:
                textTrack[j[0]] = []
            wList = []
            for i in temp1:
                #Removing URLs and numbers
                x = re.sub(r'https?\S+', '', re.sub(r'\w*\d\w*', '', i).strip()).strip()
                tok = list(tknzr.tokenize(x))
                tList = buildVocab(tok)
                wList += tList
            fVec1 = buildFeatureVector(wList)
            #fVec.append(fVec1)
            fVec = fVec1
            opVec1 = [0 for i in range(16)]
            opVec1[indexOfMbtiType(j[0])] = 1.0
            #opVec.append(opVec1)
            opVec = opVec1
            #if rownum % batchSize == 0:
            c,_ = sess.run([cost, opt], feed_dict = {ipTensor:[fVec], opTensor:[opVec]})
            '''if rownum % batchSize == 0:
                fVec = []
                opVec = []
                avgCost += c #* batchSize / nSamples
            '''
            avgCost += c/nSamples
        #if epoch % dispEpoch == 0:
        currTime = time.time()
        currTimeOnClock = datetime.datetime.now().time()
        print('Epoch:',epoch,'of cost',avgCost,'takes:',currTime - startTime,'; Time:',currTimeOnClock)
        startTime = time.time()
    
    print('Optimization done')
    
    #Testing model
    correctPrediction = tf.equal(tf.argmax(prediction,1), tf.argmax(opTensor, 1))
    
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))
    totalTestData = nSamples
    #batch_x_test,batch_y_test = get_batch(newsgroups_test,0,total_test_data)
    avgAccuracy = 0
    for j in data[nSamples:nSamples + testSize]:
        i = j[1]
        rownum += 1
        temp1 = i.split('|||')
        if j[0] not in textTrack:
            textTrack[j[0]] = []
        wList = []
        for i in temp1:
            #Removing URLs and numbers
            x = re.sub(r'https?\S+', '', re.sub(r'\w*\d\w*', '', i).strip()).strip()
            tok = list(tknzr.tokenize(x))
            tList = buildVocab(tok)
            wList += tList
        fVec = [buildFeatureVector(wList)]
        opVec = [0 for i in range(16)]
        opVec[indexOfMbtiType(j[0])] = 1.0
        opVec = [opVec]
        avgAccuracy += accuracy.eval({ipTensor: fVec, opTensor: opVec})/testSize
    
    print("Accuracy:", avgAccuracy)
    
    
    
    
    
    
    
    
    
    
    
    
