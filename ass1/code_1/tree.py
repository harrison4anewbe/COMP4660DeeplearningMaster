import numpy as np
import xml.etree.ElementTree as ET
import os
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn import tree

# Load Training data
trainlabel =np.load('train_label.npy')
trainlabel = trainlabel[:,9]
trainlabel = list(map(int,trainlabel))
traindata =pd.read_csv('Train_data.csv')
traindata = traindata.values
for i in range(len(traindata[0])):
    traindata[:,i]=(traindata[:,i]-traindata[:,i].min()) /(traindata[:,i].max()-traindata[:,i].min())
X = torch.tensor(traindata, dtype=torch.float)
Y = torch.tensor(trainlabel, dtype=torch.long)
Y = Y-1
print("data ready")

clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)
print (clf.best_score_)

testlabel =np.load('test_label.npy')
testlabel = testlabel[:,9]
testlabel = list(map(int,testlabel))
testdata =pd.read_csv('Test_data.csv')
testdata = testdata.values
for i in range(len(testdata[0])):
    testdata[:,i]=(testdata[:,i]-testdata[:,i].min()) /(testdata[:,i].max()-testdata[:,i].min())

X_test = torch.tensor(testdata, dtype=torch.float)
Y_test = torch.tensor(testlabel, dtype=torch.long)
Y_test = Y_test-1
print("Testset load finished...")
X_test,Y_test = X_test.cuda(),Y_test.cuda()

Y_pred_test = clf.predict(X_test)

_, predicted_test = torch.max(F.softmax(Y_pred_test,1), 1)

total_test = predicted_test.size(0)
correct_test = np.sum(predicted_test.data.cpu().numpy() == Y_test.data.cpu().numpy())
print()
print('Validation Accuracy: %.2f %%' % (100 * correct_test / total_test)) 