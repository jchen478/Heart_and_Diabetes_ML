#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 22:13:24 2018

@author: jingyaochen
"""
import numpy as np
import pandas as pd 

def EdgeWeight(data, meta, att1, att2, Pxd, PyArr):
    
    # initilalize constants
    nAtt1 = len(meta._attributes[att1][1])
    nAtt2 = len(meta._attributes[att2][1])
    nClass = len(meta._attributes['class'][1])
    classNames = meta._attributes['class'][1]
       
    # conditional probability P(xi,xj|y)
    Pcond = np.zeros([2, nAtt1, nAtt2])     
    for y in range(nClass):
        dataCond = data[data['class'] == classNames[y]]
        nMatch = len(dataCond)+nAtt1*nAtt2
        for j in range(nAtt2):
            xj = meta._attributes[att2][1][j]
            for i in range(nAtt1):
                xi = meta._attributes[att1][1][i]
                dataMatch = dataCond[dataCond[att1] == xi]
                dataMatch2 = dataMatch[dataMatch[att2] == xj]
                Pcond[y,i,j] = (float(len(dataMatch2))+1.0)/float(nMatch)

    # sum over xi, xj for weight I(Xi,Xj|Y)
    w = 0.0
    nTrain = len(data) + 2*nAtt1*nAtt2
    for y in range(nClass):
        data1 = data[data['class'] == classNames[y]]
        for j in range(nAtt2):
            xj = meta._attributes[att2][1][j]
            data2 = data1[data1[att2] == xj]
            for i in range(nAtt1):
                xi = meta._attributes[att1][1][i]
                data3 = data2[data2[att1] == xi]
                Pj = float(len(data3)+1.0)/float(nTrain)
                inside = Pcond[y,i,j]/(Pxd[att1][y,i]*Pxd[att2][y,j])
                w = w + Pj*np.log2(inside)  
    return w

# convert undirected graph to directed
def removeChild(G,node,par):
    Gstore = G[node].copy()
    for g in range(len(Gstore)):
        i = Gstore[g]
        if i != par:
            G[node].remove(i)
            removeChild(G,i,node)

def TAN(trainData,testData,meta):

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
    
    # weight between each pair of attributes
    I = np.zeros([nFeat, nFeat])
    for i in range(nFeat):
        att1 = meta._attrnames[i]
        for j in range(nFeat):
            att2 = meta._attrnames[j]
            if i != j:    
                I[i,j] = EdgeWeight(trainData,meta,att1,att2,Pxd,PyArr)
            else:
                I[i,j] = -1

    # find all edges connecting maximum spanning tree
    # start exploring from first attribute
    explored = [meta._attrnames[0]]
    unexplored = meta._attrnames.copy()
    unexplored.remove(explored[0])
    unexplored.remove('class')
    edges = [[] for i in range(nFeat-1)]
    
    # find edges in maximum spanning tree
    for i in range(nFeat-1):        
        maxI = -1
        e = [0, 0]        
        for ex in range(len(explored)):
            exind = meta._attrnames.index(explored[ex])
            for un in range(len(unexplored)):
                unind = meta._attrnames.index(unexplored[un])
                w = I[exind,unind]
                if (w > maxI):
                    e[0] = int(exind)
                    e[1] = int(unind)
                    maxI = w
        edges[i] = e
        unexplored.remove(meta._attrnames[e[1]])
        explored.append(meta._attrnames[e[1]])

    # create undirected graph from edges 
    Gun = [[] for i in range(nFeat)]
    for i in range(nFeat-1):
        Gun[edges[i][0]].append(edges[i][1])
        Gun[edges[i][1]].append(edges[i][0])
    
    # from undirected graph, create directed
    # each node contains only its parent       
    G = Gun.copy()
    removeChild(G,0,0)
    
    # add class as parent to all nodes
    for i in range(nFeat):
        G[i].append(nFeat)
        
    # print each node and its parent
    print(meta._attrnames[0],meta._attrnames[G[0][0]])
    for i in range(nFeat):
        if i == 0:
            continue
        print(meta._attrnames[i],meta._attrnames[G[i][0]],meta._attrnames[G[i][1]])
    print()
    
    # create probability table for each feature
    Pd = [[] for i in range(nFeat)]
    Pr = [[] for i in range(nFeat)]
    # create root table
    att = meta._attrnames[0]
    nAtt = len(meta._attributes[att][1])
    Pr[0] = np.zeros([nClass*nAtt,3])
    for y in range(nClass):
        cData = trainData[trainData['class'] == classNames[y]]
        nData = len(cData)+nAtt
        for i in range(nAtt):
            matchData = cData[cData[att] == meta._attributes[att][1][i]]
            Pr[0][y*nAtt+i,0] = i
            Pr[0][y*nAtt+i,1] = y
            Pr[0][y*nAtt+i,2] = (len(matchData)+1.0)/nData
    Pd[0] = pd.DataFrame(data=Pr[0],columns=[att,'class','p'])
    
    # create non root table
    for j in range(nFeat):
        if j == 0:
            continue
        att = meta._attrnames[j]
        nAtt = len(meta._attributes[att][1])
        par = meta._attrnames[G[j][0]]
        nPar = len(meta._attributes[par][1])
        Pr[j] = np.zeros([nClass*nAtt*nPar,4])
        for y in range(nClass):
            cData = trainData[trainData['class'] == classNames[y]]
            for p in range(nPar):
                attp = meta._attributes[par][1][p]
                mData = cData[cData[par] == attp]
                nData = len(mData)+nAtt
                for i in range(nAtt):
                    ind = y*nAtt*nPar+p*nAtt+i
                    matchData = mData[mData[att] == meta._attributes[att][1][i]]
                    Pr[j][ind,0] = i
                    Pr[j][ind,1] = p
                    Pr[j][ind,2] = y
                    Pr[j][ind,3] = (len(matchData)+1.0)/nData
        Pd[j] = pd.DataFrame(data=Pr[j],columns=[att, par, 'class','p'])
    
   
    # prediction
    nCor = 0
    for ind in testData.index:    
        
            P = PyArr.copy()
            # convert each feature in test instance to indices
            featInd = np.zeros([nFeat])
            for i in range(nFeat):
                att = meta._attrnames[i]
                featInd[i] = meta._attributes[att][1].index(testData.iloc[ind][i])

            # probability for root
            table = Pd[0]
            attInd = featInd[0]
            att = meta._attrnames[0]
            for y in range(nClass):
                m = table[(table['class'] == y) \
                          & (table[att] == attInd)]
                P[y] = P[y]*m.iloc[0][-1]

            # probability of other features
            for i in range(nFeat):
                if i == 0:
                    continue;
                attInd = featInd[i]
                att = meta._attrnames[i]
                parInd = G[i][0]
                par = meta._attrnames[parInd]
                table = Pd[i]
                for y in range(nClass):
                    m = table[(table['class'] == y) \
                          & (table[att] == attInd) \
                          & (table[par] == featInd[parInd])]
                    P[y] = P[y]*m.iloc[0][-1]
            P = P[0]/(P[0]+P[1])
            if P >= 0.5:
                predicted = meta._attributes['class'][1][0]
            else:
                P = 1-P
                predicted = meta._attributes['class'][1][1]
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