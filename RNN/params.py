import torch

learning_rate = 0.5
embedding_size = 300
hidden_size = 50
sentence_length = 10
batch_size = 32
epochs = 10
classes = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
