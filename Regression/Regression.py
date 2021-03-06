

##Yo yo welcome to a little regression lib
##There are certainly places I could make this more compact.... I like for loops ok!??
#Why use functional programming when few loop do trick?

#Author: Matt Collyer, matthewcollyer@bennington.edu
#For Machine Learning, Bennington, 2021

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def graph(X,Y, W):
  #Scatter plots X and Y, and graphs function with weights W
  x = np.linspace(X.min(), X.max(),100)
  y = 0
  for i in range(len(W)):
    y += W[i]* (x**i)
  plt.plot(x,y, 'g')
  plt.scatter(X,Y)
  plt.show()



def normalize(dataframe):
  #Normalizes data by compressing range to 0 - 1
  data = {}
  for column in dataframe:
    min_x = min(dataframe[column])
    max_x = max(dataframe[column])
    normalized_column = []
    for value in dataframe[column]:
      normalized_column.append((value - min_x)/(max_x - min_x))
    data[column] = normalized_column
  return data


def higher_order(V, degree):
  #Accepts vector. gives matrix of x up to degree. 
  # X, X^2, X^3 .. .etc.
  #ONLY ACCEPTS DEGREES 2 OR HIGHER
  if(degree < 2):
    return V
  X = V
  for i in range(2,degree+1):
    X = np.column_stack((X, np.power(V,i)))
  return X

def cost(W, X, Y):
  #calculates cost -- half the sum of the errors squared.
  y_hat = X @ W
  errors = y_hat - Y
  return sum([e**2 for e in errors])/2


def gradient_descent(W, X, Y, tolerance, iterations, beta):
  #Where the magic happens. 
  #Returns Weights, Steps taken, MSE (RSS/N)
  #Uses backtracking line search to find step size.
  for t in range(1, iterations+1):
    step_size = 1.0
    derivative_weights = np.zeros(X.shape[0]-1)
    y_hat = X @ W
    errors = y_hat - Y
    derivative_weights = X.T @ errors
    gradient_mag_squared = np.dot(derivative_weights, derivative_weights)
    current_cost = cost(W, X, Y)
    while( cost(W - derivative_weights * step_size, X, Y) > (current_cost - ((step_size/2) * gradient_mag_squared))) :
      step_size *= beta
    W = W - derivative_weights * step_size
    if((t >= iterations) or (gradient_mag_squared < tolerance ** 2)):
      return({'Weights': W, 'Steps':t, 'MSE':sum([e**2 for e in errors])/X.shape[0]})



def calculate_MSE(W,X,Y):

  errors = (X @ W) - Y
  return sum([e**2 for e in errors])/X.shape[0]


def lowest_mse(descents):
  #Given a list of dicts of descents, returns the dict with the lowest MSE
  mses = []
  for descent in descents:
    mses.append(descent['MSE'])
  return descents[mses.index(min(mses))]

def raise_variables(original_X, degree):
  #This is just a dumb function so I can raise have multiple variables raised to a degree.
  for i in range(original_X.shape[1]):
    variable = higher_order(original_X[:,i], degree)
    if(i == 0):
      X = variable
    else:
      X = np.column_stack((variable, X))
  return X


def polynomial_regression(original_X, Y, limit = 8, iteration_max = 200000, tolerance = 0.01, beta = 0.8, updates = True):
  descents = []
  for i in range(1, limit+1):
    if(len(original_X.shape)>1):
      X = raise_variables(original_X, i)
    else:
      X = higher_order(original_X, i)
    X = np.column_stack((np.ones(X.shape[0]), X))
    W = np.zeros(X.shape[1])
    descent = gradient_descent(W, X, Y, tolerance, iteration_max, beta)
    descent['degree'] = i
    descents.append(descent)
    if(updates):
      print("Variables with degree ",i, "finished in ", descent['Steps'], " steps with an MSE of ", descent['MSE'])
  return descents

