import numpy as np


###################################################################
# Read the data
#   - There are two data files, one negative and one positive
#   - Each line is a new sentence
#   - Each token is seperated by a space
#   - Write a function that will take in a positive file and a negative file,
#     and return a list of sentences (lists of tokens)
#            and a list of labels (1 for positive, 0 for negative)


def words_from_file(positive_file, negative_file):
  labels = []
  words = []
  with open(positive_file) as pfile:
    for line in pfile:
      words.append(line.split())
      labels.append(1)
  with open(negative_file) as nfile:
    for line in nfile:
      words.append(line.split())
      labels.append(0)
  return (words, labels)


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
  word_id = 1
  dictionary = {word: word_id for(word, value) in dictionary.items() if value > 2}
  dictionary['<oov>'] = 0
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
  for word in sentence:
    if(word not in vocab):
      new_sentence.append(0)
    else:
      new_sentence.append(vocab[word])
  return new_sentence

def sentences_to_ints(sentences,vocab):
  encoded = []
  for sentence in sentences:
    encoded.append(sentence_to_ints(sentence,vocab))
  return encoded


# ##################################################################
# Vectorize the sentences using Bag of Words
#  - Turn each sentence into a vector of V dimensions
#    where V is the size of the vocabulary dictionary
#  - Each element of the vector should be the count of the words in the sentence,
#    that correspond to that index

def bag_of_words_vector(sentence,vocab_length):
  vector = np.zeros(vocab_length)
  for encoded_word in sentence:
    vector[encoded_word] += 1
  return vector



###################################################################
# Put it all together
#  - Read the examples in from the file
#  - construct the vocabulary for the corpus
#  - convert the examples into lists of ints
#  - return the tuple (X, Y), where X is a numpy matrix of integers num_samples x vocab_size
#      and Y is a vector of 0 or 1s

#this now takes in both training and test to make them the same size + using same corpus

def load_and_preprocess_data(positive_train, negative_train, positive_test, negative_test):
  sentences, sentiment = words_from_file(positive_train, negative_train)
  vocab = build_vocabulary(sentences)
  encoded_sentences = sentences_to_ints(sentences,vocab)
  vocab_length = len(vocab)
  X =[]
  for sentence in encoded_sentences:
    X.append(bag_of_words_vector(sentence, vocab_length))


  #Get test data the same size with same vocab as training
  test_sentences, test_sentiment = words_from_file(positive_test, negative_test)
  test_X = []
  for sentence in sentences_to_ints(test_sentences, vocab):
    test_X.append(bag_of_words_vector(sentence,vocab_length))

  for extra in range(len(X) - len(test_X)):
    test_X.append(np.zeros(len(X[0])))
    test_sentiment.append(0)

  Y = np.array((sentiment))
  X = np.array((X))
  X = np.column_stack((np.ones(X.shape[0]), X))
  W = np.zeros(X.shape[1])
  
  test_X = np.array((test_X))
  test_X = np.column_stack((np.ones(test_X.shape[0]), test_X))
  test_Y = np.array((test_sentiment))

  return{'X':X, 'Y': Y, 'W':W, 'test x': test_X, 'test y':test_Y}



if(__name__ == '__main__'):
  load_and_preprocess_data('Data/pos_sample.txt', 'Data/neg_sample.txt','Data/pos_test.txt', 'Data/neg_test.txt')
