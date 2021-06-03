from preprocess import get_preprocessed_data
from params import *

import torch
from torch import nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, dict_size, embedding_size, hidden_size, classes):
        super(RNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Embedding(dict_size, embedding_size),
            nn.ReLU(),
            nn.RNN(embedding_size, hidden_size)

        )
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, classes)
        )    

    def forward(self, x):
        x, _ = self.layers(x)
        x = x[:, -1, :]
        x = self.linear(x)
        pred = F.softmax(x, dim=1)
        return pred



def train(dataloader, model, loss_fn, optimizer, epochs):
  for epoch in range(0, epochs):
    print("Epoch", epoch)
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


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if(__name__ == '__main__'):

  data = get_preprocessed_data()
  vocab_length = data['vocab_length']
  model = RNN(vocab_length, embedding_size, hidden_size, classes).to(device)
  loss = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  train(data['train'], model, loss, optimizer, epochs)
  test(data['test'], model, loss)


