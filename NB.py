#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 22:14:07 2018

@author: jingyaochen
"""
import numpy as np
import pandas as pd


def NB(trainData,testData,meta):

    # constants
    nFeat = len(meta._attrnames)-1

    # print parents for each feature
    for i in range(nFeat):
        print(meta._attrnames[i],'class')
    print()
    
    # constants
    nFeat = len(meta._attrnames)-1
    nClass = len(meta._attributes['class'][1])
    classNames = meta._attributes['class'][1]
    
    # probability of class
    PyArr = np.zeros([nClass,1])
    nTrain = len(trainData)+nClass
    for y in range(nClass):
        matchData = trainData[trainData['class'] == classNames[y]]
        PyArr[y] = (float(len(matchData))+1.0)/float(nTrain)
    
    # create a table of individual feature probabilities
    Pxd = dict()
    for i in range(nFeat):
        att = meta._attrnames[i]
        nAtt = len(meta._attributes[att][1])
        Px = np.zeros([2, nAtt])
        for y in range(nClass):         
            condData = trainData[trainData['class'] == classNames[y]]
            nData = len(condData)+nAtt
            for j in range(nAtt):
                attName = meta._attributes[att][1][j]
                matchData = condData[condData[att] == attName]
                Px[y,j] = (float(len(matchData))+1.0)/float(nData)
        Pxd.update({att:Px})
        
    # classification
    nCor = 0
    for ind in testData.index:
        numerator = PyArr[0]
        denom2 = PyArr[1]
        for i in range(nFeat):
            att = meta._attrnames[i]
            j = meta._attributes[att][1].index(testData.iloc[ind][i])
            numerator = numerator*Pxd[att][0,j]
            denom2 = denom2*Pxd[att][1,j]
        P0 = numerator/(numerator+denom2)
        if P0 >= 0.5:
            predicted = meta._attributes['class'][1][0]
            P = P0
        else:
            predicted = meta._attributes['class'][1][1]
            P = 1-P0
        testLabel = testData.iloc[ind][-1]
        if (predicted[0] == "'"):
            predicted = predicted[1:]
            predicted = predicted[:-1]
        if (testLabel[0] == "'"):
            testLabel = testLabel[1:]
            testLabel = testLabel[:-1]
        print(predicted, testLabel,"%.12f" % P)
        if predicted == testLabel:
            nCor = nCor + 1
    print()
    print(nCor)
    
    return nCor/float(len(testData))