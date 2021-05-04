import torch
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import ToTensor
from dataset import *

#Now with hidden layers and dropout!
def softmax_regression(X, W_h1, W_h2, W_o, drop_h1, drop_h2, is_training = True):
  flat_X = X.reshape((-1, W_h1[0].shape[0])) 
  score = flat_X @ W_h1[0] + W_h1[1]
  H1 = relu(score)
  if is_training:
    H1 = dropout(H1, drop_h1)
  score = H1 @ W_h2[0] + W_h2[1]
  H2 = relu(score)
  if is_training:
    H2 = dropout(H2, drop_h2)
  score = H2 @ W_o[0] + W_o[1]
  return softmax(score)

def softmax(score):
  e_X = torch.exp(score)
  sum_e = e_X.sum(axis = 1, keepdim = True)
  return e_X / sum_e

def relu(X):
  zeros = torch.zeros_like(X)
  return torch.max(zeros, X)

def dropout(X, p):
  random = torch.Tensor(X.shape).uniform_(0,1)
  mask = (random > p).float()
  return mask * X / (1 - p)

def loss(y_hat, y):
  #sum of -log(y_hat) * y
  one_hot = torch.zeros(y_hat.shape)
  y = y.unsqueeze(1)
  one_hot.scatter_(1, y, 1)
  neg_log = -torch.log(y_hat)
  return (one_hot * neg_log).sum()

def gradient_descent(layers_W, step_size, batch_size):
  with torch.no_grad():
    for layer_W in layers_W:
      for w in layer_W:
        w -= w.grad * step_size / batch_size
        w.grad.zero_()

def setup_nn(num_inputs, num_outputs, num_hidden1, num_hidden2):
 #Step 1 - initialize weights and bias
  W = torch.normal(0, 0.01, size = (num_inputs, num_hidden1), requires_grad=True)
  b = torch.zeros(num_hidden1, requires_grad=True)
  W_h1 = (W, b)
  W = torch.normal(0, 0.01, size = (num_hidden1, num_hidden2), requires_grad=True)
  b = torch.zeros(num_hidden2, requires_grad=True)
  W_h2 = (W, b)
  W = torch.normal(0, 0.01, size = (num_hidden2, num_outputs), requires_grad=True)
  b = torch.zeros(num_outputs, requires_grad=True)
  W_o = (W, b)
  return [W_h1, W_h2, W_o,]


def train(nn, data_iterator, epochs, step_size, dropout = (0.2, 0.5)):
  (W_h1, W_h2, W_o) = nn
  for _ in range(epochs):
    for X, y in data_iterator:
      #Calculate predictions ->
      y_hat = softmax_regression(X, W_h1, W_h2, W_o, dropout[0], dropout[1])
      #Calculate loss
      l = loss(y_hat, y)
      # compute gradient <-
      l.backward()
      gradient_descent([W_h1, W_h2, W_o], step_size, X.shape[0])
  return [W_h1, W_h2, W_o,]

def accuracy(data_iterator, model, loss_fn):
  (W_h1, W_h2, W_o) = model
  size = len(data_iterator.dataset)
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in data_iterator:
      y_hat = softmax_regression(X, W_h1, W_h2, W_o, 0, 0, is_training= False)
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
  nn = setup_nn(784, 10, 256, 256,)
  print("Model 3")
  model = train(nn, train_dataloader, 3, 0.1)
  accuracy(test_dataloader, model, loss)
  print("Model 4")
  model = train(nn, train_dataloader, 2, 0.01, dropout = (0.4, 0.6))
  accuracy(test_dataloader, model, loss)