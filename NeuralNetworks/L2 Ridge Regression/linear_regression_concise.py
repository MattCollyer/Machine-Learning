# Linear Regression as a neural network
import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt
def synthetic_data(w : torch.tensor, b : float, num_examples : int):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def rrloss(W, l2):
    return (l2 * (W**2)).sum()

def minibatch_iterator(inputs, output, batch_size, shuffle=True):
    dataset = data.TensorDataset(inputs, output)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle)

def train(net, optimizer, loss, data_iterator, num_epochs = 1, l2 = 0):
    loss_sum = 0
    total_samples = 0
    
    for epoch in range(num_epochs):
        for mini_X, mini_Y in data_iterator:
            Y_hat = net(mini_X)
            l = loss(Y_hat, mini_Y) + rrloss(net.weight,l2)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_sum += l
            total_samples += mini_X.shape[0]

def graph(X,Y,W, bias):
  #Scatter plots X and Y, and graphs function with weights W
  x = np.linspace(X.min(), X.max(),100)
  y = 0
  for i in range(len(W)):
    y += W[i]*(x**(i+1)) + bias
  plt.plot(x,y,'g')
  Y_ = np.column_stack((Y, Y))
  Y = np.column_stack((Y_, Y))
  plt.scatter(X,Y)
  plt.show()
            
if __name__ == '__main__':

    true_w = torch.tensor([-2.3, 1.8, 3.6])
    features, labels = synthetic_data(true_w, 5.2, 1000)
    
    #set up the nn
    linear = torch.nn.Linear(features.shape[1], 1)
    linear.weight.data.normal_(0, 0.01) #weights = torch.normal(0, 0.01, size=(m, 1), requires_grad=True)
    linear.bias.data.fill_(0) #bias = torch.zeros(1, requires_grad=True)
    
    #set up optimizer
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.03)
        
    #set up loss
    squared_loss = torch.nn.MSELoss()
    
    #set up data iterator
    data_iterator = minibatch_iterator(features, labels, 10, True)
    
    #train
    
    
    # Ideally weights should be:
    # 5.2, -2.3, 1.8, 3.6

    
    l2_lambdas = [0, 0.01, 0.1, 1, 10]
  
    for lamb in l2_lambdas:
        print("Now with Lambda as ", lamb)
        train(linear, optimizer, squared_loss, data_iterator, num_epochs=10, l2 = lamb)
        weights = linear.weight.detach().numpy()[0]
        bias = linear.bias.detach().numpy()[0]
        print(weights, bias)
        graph(features, labels, weights, bias)
