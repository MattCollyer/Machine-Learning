# kernal tacking place of weights in linear r egression . X come in Y come out 

# XW + B = yhat 

# in this case no bias. 
# The multiplication is replaced with cross correlation



# Linear Regression as a neural network

from cross_correlation import *


# Linear Regression as a neural network
import numpy as np
import torch
from torch.utils import data

def minibatch_iterator(inputs, output, batch_size, shuffle=True):
    dataset = data.TensorDataset(inputs, output)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle)

def train(net, optimizer, loss, X, Y, num_epochs = 1):
    loss_sum = 0
    total_samples = 0
    
    for epoch in range(num_epochs):
      Y_hat = cross_correlation(X, net.weight)
      l = loss(Y_hat, Y)
      optimizer.zero_grad()
      l.backward()
      optimizer.step()
      loss_sum += l

if __name__ == '__main__':


    X = torch.tensor(
       [[1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 0., 0., 1., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 1., 0., 0., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.]])
        
    Y = torch.tensor( 
       [[ 0.,  0.,  1.,  1.,  0.,  0.,  0.],
        [ 0.,  1.,  1.,  0.,  0.,  0.,  0.],
        [ 1.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0.,  0., -1., -1.,  0.],
        [ 0.,  0.,  0., -1., -1.,  0.,  0.]])

    #set up the nn
    linear = torch.nn.Linear(2,2)
    linear.weight.data.normal_(0, 0.01) #weights = torch.normal(0, 0.01, size=(m, 1), requires_grad=True)
    linear.bias.data.fill_(0) #bias = torch.zeros(1, requires_grad=True)
    
    #set up optimizer
    optimizer = torch.optim.SGD(linear.parameters(), lr=1e-2)
        
    #set up loss
    squared_loss = torch.nn.MSELoss()
    
    #train
    
    train(linear, optimizer, squared_loss, X, Y, num_epochs=100)
    print(linear.weight)