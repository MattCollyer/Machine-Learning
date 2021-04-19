

##Yo yo welcome to a little regression gradient ascent lib
##There are certainly places I could make this more compact.... I like for loops ok!??
#This is late and should totally be documented more. I KNOW. 
#Author: Matt Collyer, matthewcollyer@bennington.edu
#For Machine Learning, Bennington, 2021


import math
import numpy as np
import matplotlib.pyplot as plt

def preprocess(X):
  #Simple preprocessing for test data.
    X = np.column_stack((np.ones(X.shape[0]), X))
    W = np.zeros(X.shape[1])
    return{'X':X, 'Weights':W}

def sigmoid(score):
  return ((1 + (math.e ** -score)) ** -1)

def sigmoids(scores):
   return [sigmoid(score) for score in scores]

def batch_gradient_ascent(W, X, Y, tolerance, iterations, step_size, updates = True):
  for i in range(1, iterations+1):
      if(updates and i % 100 == 0):
        print("Step:",i)
      scores = X @ W
      probabilities = sigmoids(scores)
      partial_derivatives = X.T @ (Y - probabilities)
      W += step_size * partial_derivatives
      if(sum(np.square(partial_derivatives)) < tolerance ** 2):
        break
  return {'Weights': W, 'Steps Taken': i-1}

def shuffle(x, y):
  shuffled = np.column_stack((x, y))
  np.random.shuffle(shuffled)
  X = shuffled[:, 0: x.shape[1]]
  Y = shuffled[:, x.shape[1]]
  return X, Y

def stochastic_gradient_ascent(W, X, Y, num_epochs, step_size):
  X, Y = shuffle(X,Y)
  for epoch in range(num_epochs):
    for i in range(X.shape[0]):
      score = W.T * X[i]
      y_hat = sigmoid(score)
      error = Y[i] - y_hat
      partial_derivatives = X[i] * error
      W += step_size * partial_derivatives
  return {'Weights': W, 'epochs': num_epochs}

def fscore(X, Y, W):
  counts = {'TP': 0, 'FP': 0, 'FN':0, 'TN': 0}
  for i in range(len(X)):
    predict = round(sigmoid(sum(X[i] * W)))
    actual = Y[i]
    if(int(predict) == int(actual)):
      if(int(actual) == 0):
        counts['TN'] += 1 
      else:
        counts['TP'] += 1
    else:
      if(int(actual) == 0):
        counts['FN'] += 1
      else:
        counts['TN'] += 1
  recall = counts['TP'] / (counts['TP'] + counts['FN'])
  precision = counts['TP']/ (counts['TP'] + counts['FP'])
  return 2 * (precision * recall) /(precision + recall)



if (__name__ == '__main__'):
  original_X = np.array([[9, 8],
                [5, 7],
                [8, 7],
                [4, 8],
                [0, 1],
                [9, 2],
                [8, 4],
                [4, 4],
                [3, 2],
                [3, 0]])
  Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
  step_size = 0.05
  tolerance = 0.1
  processed_data = preprocess(original_X)
  X = processed_data['X']
  W = processed_data['Weights']

  sascent = stochastic_gradient_ascent(W, X, Y, 10, step_size)
  bascent = batch_gradient_ascent(W,X,Y,tolerance, 1000, step_size)
  print(fscore(X, Y, sascent['Weights']))
