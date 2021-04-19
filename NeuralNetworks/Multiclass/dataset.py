import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def show_data():
  figure = plt.figure(figsize=(8, 8))
  cols, rows = 3, 3
  for i in range(1, cols * rows + 1):
      sample_idx = torch.randint(len(training_data), size=(1,)).item()
      img, label = training_data[sample_idx]
      figure.add_subplot(rows, cols, i)
      plt.title(labels_map[label])
      plt.axis("off")
      plt.imshow(img.squeeze(), cmap="gray")
  plt.show()


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

  labels_map = {
      0: "T-Shirt",
      1: "Trouser",
      2: "Pullover",
      3: "Dress",
      4: "Coat",
      5: "Sandal",
      6: "Shirt",
      7: "Sneaker",
      8: "Bag",
      9: "Ankle Boot",
  }

  train = DataLoader(training_data, batch_size=64, shuffle=True)
  test = DataLoader(test_data, batch_size=64, shuffle=True)
  return train, test
  
if __name__ == '__main__':



  show_data()

