import numpy as np
import xml.etree.ElementTree as ET
import os
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn import naive_bayes

# Load Training data
trainlabel =np.load('train_label.npy')
trainlabel = trainlabel[:,9]
trainlabel = list(map(int,trainlabel))
traindata =pd.read_csv('Train_data.csv')
traindata = traindata.values
for i in range(len(traindata[0])):
    traindata[:,i]=(traindata[:,i]-traindata[:,i].min()) /(traindata[:,i].max()-traindata[:,i].min())

cls=naive_bayes.MultinomialNB()
cls.fit(traindata,trainlabel)
print("training score:%.2f"%(cls.score(traindata,trainlabel)))

testlabel =np.load('val_label.npy')
testlabel = testlabel[:,9]
testlabel = list(map(int,testlabel))
testdata =pd.read_csv('val_data.csv')
testdata = testdata.values
for i in range(len(testdata[0])):
    testdata[:,i]=(testdata[:,i]-testdata[:,i].min()) /(testdata[:,i].max()-testdata[:,i].min())

print("testing score:%.2f"%(cls.score(testdata,testlabel)))