#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 14:42:15 2018

@author: jingyaochen
"""
import pip
pip.main(['install','anytree'])
pip.main(['install','pandas'])
pip.main(['install','math'])
pip.main(['install','sys'])
pip.main(['install','scipy.io'])
pip.main(['install','sklearn'])

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math 
import sys
from scipy.io import arff
from  sklearn import preprocessing 
from anytree import AnyNode, RenderTree, PreOrderIter

# train tree
def MakeSubtree(data,meta,m,par,labelDict):
    
    # entropy in data
    H = Entropy(data)    
    # data dimension
    nData = data.shape[0]    
    # check for stopping criteria
    # 1. fewer than m
    if nData < m:
        return
    # 2. all instances have the same class
    if H == 0:
        return
    # find candidate split and best info gain for each attribute
    C,candAtt = FindCandSplit(data,meta,H,labelDict)
    bestAtt = BestSplit(data,C,candAtt)
    
    # 3. no split yield info gain
    if C[bestAtt][1] <= 0.0:
        return
    # split according to type of best attribute
    if meta._attributes[bestAtt][0] == 'numeric':
        criterion = C[bestAtt][0]
        dfLeft = data[data[bestAtt] <= criterion]
        dfRight = data[data[bestAtt] > criterion]         
        probArr=Probability(dfLeft)
        nodeName = bestAtt+" <= "+str(criterion) + "[" + str(probArr[1])+ " " + str(probArr[2]) + "]:"
        Nleft = AnyNode(id=nodeName, parent=par, att=bestAtt, crit=C[bestAtt][0], prob=probArr[0], cond='le', nsample=dfLeft.shape[0],pred='')
        probArr=Probability(dfRight)
        nodeName = bestAtt+" > "+str(criterion) + "[" + str(probArr[1])+ " " + str(probArr[2]) + "]:"
        Nright = AnyNode(id=nodeName, parent=par, att=bestAtt, crit=C[bestAtt][0], prob=probArr[0],cond='gt', nsample=dfRight.shape[0],pred='')
        MakeSubtree(dfLeft,meta,m,Nleft,labelDict)
        MakeSubtree(dfRight,meta,m,Nright,labelDict)
    elif meta._attributes[bestAtt][0] == 'nominal':
        counter = 0
        for i in labelDict[bestAtt]:
            dfB = data[data[bestAtt] == i]
            nDfB = dfB.shape[0]
            probArr=Probability(dfB)
            nodeName = bestAtt+"="+meta._attributes[bestAtt][1][counter] + "[" + str(probArr[1])+ " " + str(probArr[2]) + "]:"
            Nbranch = AnyNode(id=nodeName, parent=par, att=bestAtt, crit=meta._attributes[bestAtt][1][counter],prob=probArr[0],cond='eq',nsample=str(nDfB),pred='')
            MakeSubtree(dfB,meta,m,Nbranch,labelDict)
            counter = counter + 1
            
# check if entry in array are identical
def checkEqual(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

# Find candidate split
def FindCandSplit(data,meta,H,labelDict):    
    # find attributes with candidate split
    candAtt = meta._attrnames[:]
    candAtt.remove('class')
    for cand in candAtt:
        if checkEqual(data[cand][:]) == True:
            candAtt.remove(cand)
    # dictionary for best split for each attribute
    C = dict()
    # sort based on attribute
    for cand in candAtt:  
        if meta._attributes[cand][0] == 'numeric':
            data = data.sort_values(by=cand)
            uniqueList = []
            valList = []
            for ind in data.index:
                val = data[cand][ind]
                if val in uniqueList:
                    continue
                else:
                    tempData = data[data[cand] == val]
                    avg = sum(tempData.iloc[:,-1].values)/float(len(tempData))
                    uniqueList.append(val)
                    valList.append(avg)

            for i in range(len(uniqueList)):
                if i == 0:
                    bestInfoGain = -1000.0
                    bestMid = 0.0
                if i != 0:
                    val = valList[i]
                    pval = valList[i-1]
                    if val != pval:
                        mid = 0.5*(uniqueList[i]+uniqueList[i-1])
                        dfLeft = data[data[cand] <= mid]
                        dfRight = data[data[cand] > mid]
                        Hleft = Entropy(dfLeft)
                        Hright = Entropy(dfRight)
                        Hcond = float(dfLeft.shape[0])/data.shape[0]*Hleft+float(dfRight.shape[0])/data.shape[0]*Hright
                        infoGain = H - Hcond
                        if infoGain > bestInfoGain:
                            bestMid = mid
                            bestInfoGain = infoGain
            C.update({cand:[bestMid,bestInfoGain]})
        elif meta._attributes[cand][0] == 'nominal':
            Hcond = 0.0
            nData = float(data.shape[0])
            for i in labelDict[cand]:
                dfB = data[data[cand] == labelDict[cand][i]]
                HB = Entropy(dfB)
                Hcond = Hcond + dfB.shape[0]/nData*HB
            C.update({cand:[meta._attributes[cand][1],H-Hcond]})    
    return C, candAtt

# Find the best split
def BestSplit(data,C,candAtt):
    bestInfoGain = -1000
    bestAtt = ''
    for cand in candAtt:
        if C[cand][1] > bestInfoGain:
            bestAtt = cand
            bestInfoGain = C[cand][1]
    return bestAtt

# Find the entropy in data
def Entropy(data):
    nData = float(len(data))
    if nData == 0:
        return 0.0
    nPos = sum(data.iloc[:,-1].values)
    if nPos == 0:
        return 0.0
    nNeg = nData - nPos
    if nNeg == 0:
        return 0.0
    H = - float(nPos)/nData*math.log(nPos/nData,2)\
        -float(nNeg)/nData*math.log(nNeg/nData,2)
    return H

# Find the probability of the positive class
def Probability(data):
    if len(data) == 0:
        return np.array([0.5, 0., 0.])
    nPos = sum(data.iloc[:,-1].values)
    nNeg = len(data) - nPos
    if nPos == 0:
        return np.array([0., nNeg, nPos])
    if nNeg == 0:
        return np.array([1., nNeg, nPos])
    return np.array([nPos/float(len(data)), nNeg, nPos])

# Find the parent probability
def FindParentProb(tree):
    if tree.parent.prob != 0.5:
        return tree.parent.prob
    return FindParentProb(tree.parent)

# Assign prediction at leaf nodes
def AssignPrediction(tree):
    for node in PreOrderIter(tree):
        if node.is_leaf is True:
            if node.prob < 0.5:
                node.pred = 'negative'
            elif node.prob > 0.5:
                node.pred = 'positive'
            elif node.prob == 0.5:
                probability = FindParentProb(node)
                if probability < 0.5:
                    node.pred = 'negative'
                elif probability > 0.5:
                    node.pred = 'positive'
                    
# Predict class for test instance 
def PredictClass(instance,tree,meta):
    if tree.is_leaf == True:
        return tree.pred
    for node in tree.children:
        if node.cond == 'gt':
            if instance[node.att] > node.crit:
                return PredictClass(instance,node,meta)
        elif node.cond == 'le':
            if instance[node.att] <= node.crit:
                return PredictClass(instance,node,meta)
        elif node.cond == 'eq':
            if instance[node.att] == node.crit:
                return PredictClass(instance,node,meta)
    return -1

#%% Main Program   

# input parameters
m = int(sys.argv[1]) # stopping criterion
trainFilename = sys.argv[2]
testFilename = sys.argv[3]

dataset, meta = arff.loadarff(trainFilename)
data = pd.DataFrame(dataset)

# transform nominal attributes to label
labelDict = dict() # dictionary for labels
for row in meta:
    dtype = meta._attributes[row]
    if dtype[0] == 'nominal':
        data[row] = data[row].str.decode('utf-8')
        nom = preprocessing.LabelEncoder()
        lab = nom.fit_transform(meta._attributes[row][1])
        labelDict.update({row: lab})
        data[row] = nom.transform(data[row])

# train tree
dataProb = Probability(data)
if dataProb[0] == 0.5:
    dataProb[0] = 1
tree = AnyNode(id='root',prob=dataProb[0])
MakeSubtree(data,meta,m,tree,labelDict)

# walk through nodes and assign predictions
AssignPrediction(tree)
  
#%% Predict Class For Test Instances     
dataTest, metaTest = arff.loadarff(testFilename)
# preprocessing, decode strings
dataT = pd.DataFrame(dataTest)
for row in metaTest:
    dtype = metaTest._attributes[row]
    if dtype[0] == 'nominal':
        dataT[row] = dataT[row].str.decode('utf-8')
strs = ["" for x in range(len(dataT))]
for ind in range(len(dataT)):
    strs[ind] = PredictClass(dataT.iloc[ind][:],tree,metaTest)
dataT['predicted class'] = pd.Series(strs, index=dataT.index)

#%% Decision Tree Output
for pre, fill, node in RenderTree(tree):
    if node.is_leaf is True:
        print("%s%s%s" %(pre, node.id,node.pred))
    else:
        print("%s%s" %(pre, node.id))
print("<Predictions for the Test Set Instances>")
for ind in range(len(dataT)):
    print(ind+1, ": Actual:", dataT['class'][ind]," Predicted: ", dataT['predicted class'][ind])
nCor = 0
for ind in range(len(dataT)):
    if dataT['class'][ind] == dataT['predicted class'][ind]:
        nCor = nCor + 1
print("Number of correctly classified:",nCor, "Total number of test instances:",len(dataT))
