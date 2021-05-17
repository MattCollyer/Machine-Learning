import torch
learning_rate = 0.8
batch_size = 128
epochs = 10
classes = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
