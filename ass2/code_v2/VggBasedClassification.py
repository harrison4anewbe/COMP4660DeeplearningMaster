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
######################################################### LOAD DATA #########################################################
class PatchDataset(Dataset): # define read data
    def __init__(self, data_dir, transform=torchvision.transforms.ToTensor()):
        """
        :param data_dir: dataset directory
        :param transform: data preprocessing
        """
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform # normalization and transfer to tensor
        self.transform2 = torchvision.transforms.RandomHorizontalFlip() # randomly flip
    def __getitem__(self, item):
        path_img, label = self.data_info[item]
        image = Image.open(path_img).convert('RGB')
        # Transform
        if self.transform is not None:
            image = self.transform2(image)
            image = self.transform(image)
            
        return image, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
#         path_dir = os.path.join(data_dir, 'train_dataset.npy')
        return np.load(data_dir)
    
train_dataset = PatchDataset('./train_dataset.npy')
test_dataset = PatchDataset('./test_dataset.npy')
val_dataset = PatchDataset('./val_dataset.npy')

train_loader = torch.utils.data.DataLoader( dataset=train_dataset, # load data, batchsize 256, random order
    batch_size=256, shuffle=True, drop_last = True )
test_loader = torch.utils.data.DataLoader( dataset=test_dataset,
    batch_size=256, shuffle=True, drop_last = True )
val_loader = torch.utils.data.DataLoader( dataset=val_dataset,
    batch_size=256, shuffle=True, drop_last = True )

print(len(train_loader))
print(len(test_loader))

######################################################### DEFINE NETWORK #########################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 48,kernel_size=11,stride=4,padding=2)# kernel 11 stride 4 padding 2 input[3, 256, 256] output[48, 63, 63]
        self.pool = nn.MaxPool2d(3, 2)                                 # kernel 3 stride 2 input[48, 63, 63] output[48, 31, 31]
        self.conv2 = nn.Conv2d(48, 128, 5, 1, 2)                       # kernel 5 stride 1 padding 2 output[128, 31, 31]
        self.conv3 = nn.Conv2d(128, 192, 3, 1, 1)                      # kernel 3 stride 1 padding 1 output[192, 15, 15]
        self.conv4 = nn.Conv2d(192,192, 3, 1, 1)                       # kernel 3 stride 1 padding 1 output[192, 15, 15]
        self.conv5 = nn.Conv2d(192,128, 3, 1, 1)                       # kernel 3 stride 1 padding 1 output[128, 15, 15]
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*7*7, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1362)
        self._initialize_weights()
        
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.pool(F.leaky_relu(self.conv5(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x): # reshape the tensor
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    def _initialize_weights(self): # init the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):                            
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)                    
            elif isinstance(m, nn.Linear):            
                nn.init.normal_(m.weight, 0, 0.01)    
                nn.init.constant_(m.bias, 0)
                
######################################################### TRAIN NETWORK #########################################################
alexnet = Net() # init the network
loss_func = torch.nn.CrossEntropyLoss() # define loss function
learning_rate = 0.0003 # define learning rate
optimiser = torch.optim.Adam(alexnet.parameters(), lr=learning_rate) # define optimiser
all_losses = [] # store all losses for visualisation
Iteration = 50 # init iteration
it,it2 = 0,0 # init cnt

use_gpu = torch.cuda.is_available() # use GPU is available
if(use_gpu):
    alexnet = alexnet.cuda()
    loss_func = loss_func.cuda()

# train a neural network
for time in range(Iteration):
    alexnet.train() # train the network
    it =0
    total,correct = 0,0
    for batch,label in train_loader:
        # Change data type
        batch = torch.Tensor(batch).float()
        
        label = list(map(int, label))
        label = np.array(label)
        label = torch.Tensor(label).long()

        X_batch = Variable(batch, requires_grad=True).float()
        y_batch = Variable(label).long()
        # CUDA
        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
        
        # Clear the gradients before running the backward pass.
        optimiser.zero_grad()
        # Perform forward pass: compute predicted y by passing x to the model.
        Y_pred = alexnet(X_batch)
        # Compute loss
        loss = loss_func(Y_pred, y_batch)
        
        # CPU
        it += 1
        if it%50 ==0:
            loss = loss.cpu()
            print('Batch [%d/%d] Loss: %.4f' % (it + 1, len(train_loader), loss.data))
        
        _, predicted = torch.max(Y_pred, 1)
        # calculate and print accuracy
        total += predicted.size(0)
        correct += sum( predicted.data.cpu().numpy() == label.data.cpu().numpy() )
        
        # Perform backward pass
        loss.backward()
        # Calling the step function on an Optimiser makes an update to its parameters
        optimiser.step()
    # write down the loss
    all_losses.append(loss.item())
    print('Epoch [%d/%d]  Accuracy: %.2f %%' % (time + 1, Iteration, 100 * correct/total))
    # test the accuracy each 5 training
    if (time+1)%5 ==0:
        with torch.no_grad():
            total,correct = 0,0
            for batch,label in test_loader:
                # Change data type
                batch = torch.Tensor(batch).float()
                label = list(map(int, label))

                label = np.array(label)
                label = torch.Tensor(label).long()

                X_batch = Variable(batch, requires_grad=True).float()
                y_batch = Variable(label).long()

                # CUDA
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()
                Y_pred = alexnet(X_batch)

                # Compute loss
                loss = loss_func(Y_pred, y_batch)

                # CPU            
                _, predicted = torch.max(Y_pred, 1)
                # calculate and print accuracy
                total += predicted.size(0)
                correct += sum( predicted.data.cpu().numpy() == label.data.cpu().numpy() )
            print('Testset Epoch [%d/%d]  Accuracy: %.2f %%' % (time + 1, Iteration, 100 * correct/total))
# show the loss plot
plt.figure()
plt.plot(all_losses)
plt.title('Lr 0.0003')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.show()