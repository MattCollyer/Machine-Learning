from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from params import *

def get_data():
  training_data = datasets.FashionMNIST(
      root="data",
      train=True,
      download=True,
      transform=ToTensor()
  )

  test_data = datasets.FashionMNIST(
      root="data",
      train=False,
      download=True,
      transform=ToTensor()
  )

  train = DataLoader(training_data, batch_size=batch_size, shuffle=True)
  test = DataLoader(test_data, batch_size=batch_size, shuffle=True)
  return train, test