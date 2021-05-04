import torch
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import ToTensor
from dataset import *


def softmax_regression(X, W, b):
  score = X.reshape((-1, W.shape[0])) @ W + b
  return softmax(score)


def softmax(score):
  e_X = torch.exp(score)
  sum_e = e_X.sum(axis = 1, keepdim = True)
  return e_X / sum_e

def loss(y_hat, y):
  #sum of -log(y_hat) * y
  one_hot = torch.zeros(y_hat.shape)
  y = y.unsqueeze(1)
  one_hot.scatter_(1, y, 1)
  neg_log = -torch.log(y_hat)
  return (one_hot * neg_log).sum()

def gradient_descent(W, step_size, batch_size):
  with torch.no_grad():
    for w in W:
      w -= w.grad * step_size / batch_size
      w.grad.zero_()

def train(num_inputs, num_outputs, data_iterator, epochs, step_size):
  #Step 1 - initialize weights and bias
  W = torch.normal(0, 0.01, size = (num_inputs, num_outputs), requires_grad=True)
  b = torch.zeros(num_outputs, requires_grad=True)
  for _ in range(epochs):
    for X, y in data_iterator:
      #Calculate predictions ->
      y_hat = softmax_regression(X, W, b)
      #Calculate loss
      l = loss(y_hat, y)
      # compute gradient <-
      l.backward()
      gradient_descent([W, b], step_size, X.shape[0])
  return [W, b]

def accuracy(data_iterator, model, loss_fn):
  (W, b) = model
  size = len(data_iterator.dataset)
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in data_iterator:
      y_hat = softmax_regression(X, W, b)
      test_loss += loss_fn(y_hat, y).item()
      correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
  #load data
  train_dataloader, test_dataloader = get_data()
  #data labels 
  batch_size = 256
  print("Manual model 1")
  model = train(784, 10, train_dataloader, 3, 0.1)
  accuracy(test_dataloader, model, loss)
  print("Manual model 2")
  model = train(784, 10, train_dataloader, 2, 0.01)
  accuracy(test_dataloader, model, loss)
