import numpy as np
import xml.etree.ElementTree as ET
import os
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn import tree

path = './vehicle-x/train/'
trainList = os.listdir(path)
FFF,cnt = True,0
for i in trainList:
    tmp = path+i
    loadData=np.load(tmp)
    loadData = loadData.reshape(1,len(loadData))
    if FFF:
        Train_data = loadData.copy()
        FFF = False
    Train_data = np.row_stack((Train_data,loadData))
    cnt+=1
    if cnt%500 == 0:
        print(cnt)
np.savetxt('Train_data.csv', Train_data, delimiter = ',')

path = './vehicle-x/test/'
testList = os.listdir(path)
FFF,cnt = True,0
for i in testList:
    tmp = path+i
    loadData=np.load(tmp)
    loadData = loadData.reshape(1,len(loadData))
    if FFF:
        Test_data = loadData.copy()
        FFF = False
    Test_data = np.row_stack((Test_data,loadData))
    cnt+=1
    if cnt%500 == 0:
        print(cnt)
np.savetxt('Test_data.csv', Test_data, delimiter = ',')

path = './vehicle-x/val/'
valList = os.listdir(path)
FFF,cnt = True,0
for i in valList:
    tmp = path+i
    loadData=np.load(tmp)
    loadData = loadData.reshape(1,len(loadData))
    if FFF:
        val_data = loadData.copy()
        FFF = False
    val_data = np.row_stack((val_data,loadData))
    cnt+=1
    if cnt%500 == 0:
        print(cnt)
np.savetxt('val_data.csv', val_data, delimiter = ',')

root = ET.parse('./vehicle-x/finegrained_label.xml').getroot()
trainList = os.listdir('./vehicle-x/train/')
testList = os.listdir('./vehicle-x/test/')
valList = os.listdir('./vehicle-x/val/')
trainlabel,testlabel,vallabel,label = [],[],[],[]
for i in root:
    for j in i:
        tmp=[j.attrib['camDis'],j.attrib['camHei'],j.attrib['cameraID'],j.attrib['colorID'],j.attrib['imageName'], j.attrib['lightDir'],j.attrib['lightInt'],j.attrib['orientation'],j.attrib['typeID'],j.attrib['vehicleID']]
        label.append(tmp)
tmp = [x[4] for x in label]

for i in trainList:
    trainlabel.append(label[tmp.index(i)])
for i in testList:
    testlabel.append(label[tmp.index(i)])
for i in valList:
    vallabel.append(label[tmp.index(i)])

np.save('train_label.npy', trainlabel)
np.save('test_label.npy', testlabel)
np.save('val_label.npy', vallabel)