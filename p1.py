#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:00:01 2018

@author: srinidhi
"""
import csv
#import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

fil = list(csv.reader(open('mbti_1.csv')))
tknzr = TweetTokenizer()
sentAn = SentimentIntensityAnalyzer() #sentiment analyzer

stopWords = set(stopwords.words('english'))
rawtrain = [x[1] for x in fil[1:20]]
#print(rawtrain[2])
count = {}
train = {}
stopWordsFreq = {}
sentAnScores = {}

''' #Sentiment Analysis Test Code
ss = sentAn.polarity_scores(fil[1][1])
for k in ss:
    print('{0}: {1}, '.format(k, ss[k]), end='')
    print(type(k),type(ss[k]))
print()
'''

infjFile = open('esfp.csv', 'w', newline='') #enfp, enfj, esfp
writer = csv.writer(infjFile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)

for j in fil[1:]:
    i = j[1]
    ss = sentAn.polarity_scores(i)
    sst = [str(x) for x in list(ss.values())]
    
    if j[0] == 'ESFP':
        writer.writerow(','.join(sst))
    
    temp = i.split('|||')
    if j[0] not in train:
        train[j[0]] = {}
        stopWordsFreq[j[0]] = {}
        count[j[0]] = len(temp)
        #sentAnScores[j[0]] = ss
    else:
        count[j[0]] += len(temp)
       # for k in ss:
       #     sentAnScores[j[0]][k] += ss[k]
    #print(list(tknzr.tokenize(x) for x in temp))
    #tempD = {}
    for x in temp:
        tok = list(tknzr.tokenize(x))
        for y in tok:
            z = y.lower()
            if z not in stopWords:
                train[j[0]][z] = 1 if z not in train[j[0]] else train[j[0]][z]+1
                #tempD[y] = 1 if y not in tempD else tempD[y]+1
            else:
                stopWordsFreq[j[0]][z] = 1 if z not in stopWordsFreq[j[0]] else stopWordsFreq[j[0]][z]+1

for i in stopWordsFreq:
    #print(i,count[i])
    #print('Sentiments: neg={0}, neu={1}, pos={2}, compound={3}'.format(sentAnScores[i]['neg']/count[i],sentAnScores[i]['neu']/count[i],sentAnScores[i]['pos']/count[i],sentAnScores[i]['compound']/count[i]))
    '''
    if 'should' in stopWordsFreq[i]:
        print('should:',stopWordsFreq[i]['should']/count[i])
    if 'must' in train[i]:
        print('must:',train[i]['must']/count[i])
    if 'i' in stopWordsFreq[i]:
        print('i:',stopWordsFreq[i]['i']/count[i])
    if 'will' in stopWordsFreq[i]:
        print('will:',stopWordsFreq[i]['will']/count[i])
    if 'shall' in train[i]:
        print('shall:',train[i]['shall']/count[i])'''
    #print('')
    pass
    

'''
viewR = stopWordsFreq['INFJ']
for i in viewR:
    if viewR[i]>3:
        print(i,':',viewR[i])
'''