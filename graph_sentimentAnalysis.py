#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 19:16:23 2018

@author: srinidhi
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

f = open('esfp.csv').readlines()
l = []
for i in f:
    temp = i.replace('\x00','')
    l.append(temp.replace(' ',''))
fil = list(csv.reader(l))

fil = [[float(i) for i in x] for x in fil[1:]]
#print(fil)

negs = [x[0] for x in fil]
neus = [x[1] for x in fil]
poss = [x[2] for x in fil]
dnegs = gaussian_kde(negs)
dneus = gaussian_kde(neus)
dposs = gaussian_kde(poss)
xs = np.linspace(0,1)
dnegs.covariance_factor = lambda : .25
dneus.covariance_factor = lambda : .25
dposs.covariance_factor = lambda : .25
dnegs._compute_covariance()
dneus._compute_covariance()
dposs._compute_covariance()
plt.plot(xs,dnegs(xs),'r')
plt.plot(xs,dneus(xs),'g')
plt.plot(xs,dposs(xs),'b')
plt.show()