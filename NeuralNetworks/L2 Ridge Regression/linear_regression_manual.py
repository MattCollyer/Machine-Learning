# Linear Regression as a neural network
import numpy as np
import torch
from torch.utils import data
import pandas as pd 
import matplotlib.pyplot as plt
from linear_regression_nopytorch import higher_order

def synthetic_data(w : torch.tensor, b : float, num_examples : int):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
    
def minibatch_iterator(inputs, output, batch_size, shuffle=True):
    dataset = data.TensorDataset(inputs, output)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle)
    
def linear_regression_model(X, weights, bias):
    weights = weights.double()
    return X @ weights + bias
    
def squared_loss(y_hat, y):
    return (y_hat - y)**2 / 2
    
def rrloss(W, l2):
    return l2 * (W**2)

def gradient_descent(parameters, step_size, batch_size):
    with torch.no_grad():
        for param in parameters:
            param -= param.grad * step_size/batch_size
            param.grad.zero_()

def train(X : torch.tensor, Y : torch.tensor, step_size = 0.05, batch_size = 1, num_epochs = None, l2 = 0):
    m = X.shape[1]
    weights = torch.normal(0, 0.01, size=(m, 1), requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    for epoch in range(num_epochs):
        for mini_X, mini_Y in minibatch_iterator(X, Y, batch_size, True):
            Y_hat = linear_regression_model(mini_X, weights, bias)
            loss = squared_loss(Y_hat, mini_Y).sum() + rrloss(weights, l2).sum()
            loss.backward()
            gradient_descent([weights, bias], step_size, batch_size)
        with torch.no_grad():
            y_hat = linear_regression_model(X, weights, bias)
            training_loss = squared_loss(y_hat, Y)
            print('epoch', epoch, 'loss', training_loss.mean())

    return weights, bias
            

def graph(X,Y,W):
  #Scatter plots X and Y, and graphs function with weights W
  x = np.linspace(X.min(), X.max(),100)
  y = 0
  for i in range(len(W)):
    y += W[i]* (x**i)
  plt.plot(x,y, 'g')
  Y_ = np.column_stack((Y, Y))
  Y = np.column_stack((Y_, Y))
  plt.scatter(X,Y)
  plt.show()

if __name__ == '__main__':

    # true_w = torch.tensor([-2.3, 1.8, 3.6])
    # train_features, train_labels = synthetic_data(true_w, 5.2, 1000)

    # train_data = pd.read_csv('Data/demo_train.csv')
    # train_features = torch.tensor(higher_order(train_data['X'],3))
    # train_labels = torch.tensor(train_data['Y'])

    # test_data = pd.read_csv('Data/demo_test.csv')
    # test_features = torch.tensor(higher_order(test_data['X'],3))
    # test_features = torch.tensor(test_data['Y'])

    data_in = np.genfromtxt('Data/linear_regression.csv', delimiter = ',')
    train_features = torch.from_numpy(data_in[:,0:-1])
    train_labels = torch.from_numpy(data_in[:,-1])

    l2_lambdas = [0, 0.01, 0.1, 1, 10]
  
    for lamb in l2_lambdas:
        print("Now with Lambda as ", lamb)
        weights, bias = train(train_features, train_labels, step_size=0.03, batch_size = 10, num_epochs=3, l2 = lamb)


        weights = weights.detach().numpy()
        bias = bias.detach().numpy()
        W = np.row_stack((bias, weights))
        print(W)
        print(train_features.shape, train_labels.shape)
        graph(train_features, train_labels, W)


    # # Ideally weights should be:
    # # 5.2, -2.3, 1.8, 3.6
    # print('bias', bias)
    # print('weights', weights)


    