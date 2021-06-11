import numpy as np
import xml.etree.ElementTree as ET
import os
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn import tree
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn

######################################################### DATA preprocessing #########################################################
trainList = os.listdir('./ClassificationTask/train')
testList = os.listdir('./ClassificationTask/test')
valList = os.listdir('./ClassificationTask/val')

FFF,cnt = True,0

root = ET.parse('./finegrained_label.xml').getroot() # read label from xml
testlabel,vallabel,label = [],[],[]
for i in root:
    for j in i:
        tmp=[j.attrib['camDis'],j.attrib['camHei'],j.attrib['cameraID'],j.attrib['colorID'],j.attrib['imageName'], j.attrib['lightDir'],j.attrib['lightInt'],j.attrib['orientation'],j.attrib['typeID'],j.attrib['vehicleID']]
        label.append(tmp)
tmp = [ x[4] for x in label]

trainlabel = []
for i in trainList: # data format: path + label
    trainlabel.append([ './ClassificationTask/train/'+label[tmp.index(i)][4] , int(label[tmp.index(i)][9])-1 ])
for i in testList:
    testlabel.append([ './ClassificationTask/test/'+label[tmp.index(i)][4] , int(label[tmp.index(i)][9])-1 ])
for i in valList:
    vallabel.append([ './ClassificationTask/val/'+label[tmp.index(i)][4] , int(label[tmp.index(i)][9])-1 ])

np.save('train_dataset.npy', trainlabel) # save data
np.save('test_dataset.npy', testlabel)
np.save('val_dataset.npy', vallabel)

