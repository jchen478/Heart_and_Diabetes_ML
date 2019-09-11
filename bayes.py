#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 22:11:55 2018

@author: jingyaochen
"""

from TAN import TAN
from NB import NB
from scipy.io import arff
import pandas as pd
import sys
import random
import numpy as np

# read input
def readInput(file):
    dataset, meta = arff.loadarff(file)
    data = pd.DataFrame(dataset)
    # decode all attributes in dataframe
    for att in meta._attrnames:
        data[att] = data[att].str.decode('utf-8')
    return data,meta

# define cases
trainFile = sys.argv[1]
testFile = sys.argv[2]
model = sys.argv[3]

trainData,meta = readInput(trainFile)
testData,meta = readInput(testFile)  
    
if model == "n":
    accuracy = NB(trainData,testData,meta)
elif model == "t":
    accuracy = TAN(trainData,testData,meta)
