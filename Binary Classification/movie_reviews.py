from gradient_ascent import *
from preprocess import load_and_preprocess_data


def summary(step_size, iterations, epochs, tolerance):
  #Get test and training data, accept params and run both batch and stochastic ascent
  print("Parameters: step size", step_size, 'iteration max', iterations,'stochastic epochs', epochs, 'tolerance', tolerance)
  data = load_and_preprocess_data('Data/pos_sample.txt', 'Data/neg_sample.txt','Data/pos_test.txt', 'Data/neg_test.txt')
  X = data['X']
  Y = data['Y']
  W = data['W']
  test_X = data['test x']
  test_Y = data['test y']
  results = batch_gradient_ascent(W, X, Y, step_size, iterations, tolerance)
  print("Batch:")
  print(results)
  print('fscore for training:', fscore(X, Y, results['Weights']))
  print('fscore against test set', fscore(test_X, test_Y, results['Weights']))

  results = stochastic_gradient_ascent(W, X, Y, epochs, step_size)

  print("Stochastic:")
  print(results)
  print('fscore for training:', fscore(X, Y, results['Weights']))
  print('fscore against test set', fscore(test_X, test_Y, results['Weights']))



if __name__ == '__main__':
  #Params 1
  step_size = 0.05
  iterations = 1000
  epochs = 10
  tolerance = 0.1
  summary(step_size, iterations, epochs, tolerance)
  #Params 2
  step_size = 0.03
  iterations = 2000
  epochs = 5
  tolerance = 0.1
  print("Changing params")
  summary(step_size, iterations, epochs, tolerance)

