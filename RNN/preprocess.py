import pandas as pd
from params import *
import torch
from torch.utils.data import Dataset, DataLoader



class CustomDataset(Dataset):
    def __init__(self, X, Y):
      self.labels = torch.tensor(Y)
      self.text = torch.tensor(X)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        return text, label






def words_from_file(filename):
  sentences = []
  categories = []
  file = pd.read_csv(filename)
  for line in file['Title']:
    sentences.append(line.lower().split()[:sentence_length])
  category ={'b':0, 't':1, 'e':2,'m':3}
  for line in file['Category']:
    categories.append(category[line])
  return sentences, categories


###################################################################
# Build the vocabulary
#  - the vocabulary should consist of every word that appears at least 3x in the examples
#  - each word should be associated with a unique integer id
#  - <oov> should be used for every word that only appears once and should have a single id


def build_vocabulary(sentences):
  dictionary = {}
  for sentence in sentences:
    for word in sentence:
      if(word not in dictionary):
        dictionary[word] = 1
      else:
        dictionary[word] += 1
  word_id = 2
  dictionary = {word: word_id for(word, value) in dictionary.items() if value > 1}
  dictionary['<oov>'] = 0
  dictionary['<pad>'] = 1
  for word in dictionary:
    dictionary[word] = word_id
    word_id += 1
  return dictionary



###################################################################
# Convert the array of strings to an array of ints
#  - Using the vocabulary dictionary, convert each token into the approprate key
#  - If the token is not in the dictionary, use <oov>


def sentence_to_ints(sentence, vocab):
  new_sentence = []
  for i in range(sentence_length):
    if(i >= len(sentence)):
      new_sentence.append(1)
    elif(sentence[i] not in vocab):
      new_sentence.append(0)
    else:
      new_sentence.append(vocab[sentence[i]])
  return new_sentence

def sentences_to_ints(sentences,vocab):
  encoded = []
  for sentence in sentences:
    encoded.append(sentence_to_ints(sentence,vocab))
  return encoded




#Get vocabulary from corpus.

def get_preprocessed_data():

  train_titles, train_cat = words_from_file('Data/training.csv')
  test_titles, test_cat = words_from_file('Data/test.csv')



  vocab = build_vocabulary(train_titles + test_titles)
  train_titles = sentences_to_ints(train_titles,vocab)
  test_titles = sentences_to_ints(test_titles,vocab)



  train = CustomDataset(train_titles, train_cat)
  test = CustomDataset(test_titles, test_cat)

  train = DataLoader(train, batch_size=64, shuffle=True)
  test = DataLoader(test, batch_size=64, shuffle=True)

  return {'train': train, 'test': test, 'vocab_length':len(vocab)}