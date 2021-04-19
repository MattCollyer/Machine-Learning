import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import *

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), #28 x 28 pixel image
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits





def train(X, Y, step_size = 0.05, batch_size = 1, num_epochs = None):
  m = X.shape[1]
  weights = torch.normal(0, 0.01, size=(m, 1), requires_grad=True)
  bias = torch.zeros(1, requires_grad=True)
  size = len(dataloader.dataset)
  for batch, (X, y) in enumerate(dataloader):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def squared_loss(y_hat, y):
  return (y_hat - y)**2 / 2

def manual():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = NeuralNetwork().to(device)
  train_dataloader, test_dataloader = get_data()

  ##Things we decide
  learning_rate = 1e-3 #step size
  batch_size = 64
  epochs = 10
  loss_fn = squared_loss



