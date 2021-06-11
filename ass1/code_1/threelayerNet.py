import numpy as np
import xml.etree.ElementTree as ET
import os
import torch
import torch.nn.functional as F
import pandas as pd
# Load Training data
trainlabel =np.load('train_label.npy')
trainlabel = trainlabel[:,9]
trainlabel = list(map(int,trainlabel))
traindata =pd.read_csv('Train_data.csv')
traindata = traindata.values

state = np.random.get_state()
np.random.shuffle(traindata)
np.random.set_state(state)
np.random.shuffle(trainlabel)

for i in range(len(traindata[0])):
    traindata[:,i]=(traindata[:,i]-traindata[:,i].min()) /(traindata[:,i].max()-traindata[:,i].min())
X = torch.tensor(traindata, dtype=torch.float)
Y = torch.tensor(trainlabel, dtype=torch.long)
Y = Y-1

input_neurons = traindata.shape[1]
hidden_neurons = 1600
output_neurons = np.unique(trainlabel).size
learning_rate = 0.001
num_epoch = 601

# define a neural network using the customised structure
net = torch.nn.Sequential(
    torch.nn.Linear(input_neurons, 1600),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(hidden_neurons , output_neurons),
    torch.nn.Dropout(0.3),
)
# define loss function
loss_func = torch.nn.CrossEntropyLoss()

use_gpu = torch.cuda.is_available()
if(use_gpu):
    net = net.cuda()
    loss_func = loss_func.cuda()
    X,Y = X.cuda(),Y.cuda()

# define optimiser
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []

# train a neural network
for epoch in range(num_epoch):
    Y_pred = net(X)
    # Compute loss
    loss = loss_func(Y_pred, Y.long())
    all_losses.append(loss.item())
    loss = loss.cpu()

    # print progress
    if epoch % 50 == 0:
        # convert three-column predicted Y values to one column for comparison
        _, predicted = torch.max(F.softmax(Y_pred,1), 1)

        # calculate and print accuracy
        total = predicted.size(0)

        correct = predicted.data.cpu().numpy() == Y.data.cpu().numpy()

        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epoch, loss.item(), 100 * sum(correct)/total))

    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass: compute gradients of the loss with respect to all the learnable parameters of the model.
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its
    optimiser.step()

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

Y_pred_test = net(X_test)

_, predicted_test = torch.max(F.softmax(Y_pred_test,1), 1)

total_test = predicted_test.size(0)
correct_test = np.sum(predicted_test.data.cpu().numpy() == Y_test.data.cpu().numpy())
print()
print('Validation Accuracy: %.2f %%' % (100 * correct_test / total_test)) 