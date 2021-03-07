import numpy as np
import math
import pandas as pd

import matplotlib.pyplot as plt




def power_array(X,p):
  Y = []
  for x in X:
    Y.append(x**p)
  return Y



# def normalize(v):
#   #Accepts vector, normalizes it. Returns normal vector.


def higher_order(V, degree):
  #Accepts vector. gives matrix of x up to degree. 
  # X, X^2, X^3 .. .etc.
  X = V
  for i in range(2,degree+1):
    X = np.column_stack((X, power_array(V,i)))
  return X

def cost(W, X, Y):
  y_hat = X @ W
  errors = []
  for i in range(len(Y)):
    error = y_hat[i] - Y[i]
    errors.append(error)
  return sum([e**2 for e in errors])/2


def gradient_descent(W, X, Y, tolerance, iterations):
  #No more recursion. Python is bad at that!.
  #Returns Weights, Steps taken, MSE (RSS/N)



  for t in range(1, iterations+1):
    step_size = 1.0
    derivative_weights = np.zeros(X.shape[0]-1)
    y_hat = X @ W
    errors = []
    for i in range(len(Y)):
        error = y_hat[i] - Y[i]
        errors.append(error)
    derivative_weights = np.transpose(X) @ errors
    beta = 0.8
    while(cost([W[i] - derivative_weights[i] * step_size for i in range(len(W))], X, Y) > cost(W, X, Y)-((step_size/2)*(np.linalg.norm(power_array(derivative_weights, 2))))):
      step_size *= beta
    W = [W[i] - derivative_weights[i] * step_size for i in range(len(W))]

    if((t >= iterations) or ((math.sqrt(sum(power_array(derivative_weights, 2)))) < tolerance)):
      return([W,  t, sum([e**2 for e in errors])/X.shape[0]])

data = pd.read_csv('demo_train.csv')

X = data['X']
Y = data['Y']



# def graph(X,Y,W):


#   x = np.linspace(-5,5,100)
# y = x**3
#   plt.scatter(X,Y)
# x = np.linspace(-5,5,100)



X = higher_order(X, 3)
X = np.column_stack((np.ones(X.shape[0]), X))
W = np.zeros(X.shape[1])


print(gradient_descent(W, X, Y, 0.01, 200000))

