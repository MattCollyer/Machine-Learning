import pandas as pd
from Regression import *
import numpy as np


#This is a living file. by this I mean 
# i'm always commenting things out and adding stuff so it prob wont run as is

def training():
  #I just edit this if I wanna do a specific variable... 
  data = pd.read_csv('Data/Housing/training.csv')
  data = normalize(data)
  X1 = np.array(data['LandSF'])
  X2 = np.array(data['TotalFinishedArea'])
  X = np.column_stack((X1, X2))
  Y = data['TotalAppraisedValue']
  return polynomial_regression(X, Y, limit = 8)



def testing(result):
  data = pd.read_csv('Data/Housing/test.csv')
  data = normalize(data)
  # X1 = np.array(data['LandSF'])
  X2 = np.array(data['TotalFinishedArea'])
  # original_X = np.column_stack((X1, X2))
  X = higher_order(X2, result['degree'])
  X = np.column_stack((np.ones(X.shape[0]), X))
  Y = data['TotalAppraisedValue']
  mse = calculate_MSE(result['Weights'], X, Y)
  print('MSE is', mse)
  # Y = np.column_stack((Y,Y))
  graph(X2,Y,result['Weights'])


def validation(result):
  data = pd.read_csv('Data/Housing/validation.csv')
  data = normalize(data)
  # X1 = np.array(data['LandSF'])
  X2 = np.array(data['TotalFinishedArea'])
  # original_X = np.column_stack((X1, X2))
  X = higher_order(X2, result['degree'])
  X = np.column_stack((np.ones(X.shape[0]), X))
  Y = data['TotalAppraisedValue']
  mse = calculate_MSE(result['Weights'], X, Y)
  print('MSE is', mse)
  # Y = np.column_stack((Y,Y))
  graph(X2,Y,result['Weights'])





def summary():
  results = training()
  best_result = lowest_mse(results)
  print(best_result)
  W = best_result['Weights']
  Y = np.column_stack((Y,Y))
  graph(X, Y, W)




if(__name__ == '__main__'):




  #RESULTS I GOT FROM SUMMARY

  #Best training both variables
  both_result = {'Weights': [ 1.07878098e-03,  8.11974341e-01, -5.87133491e-01,  6.59019988e-01,
          1.12655324e+00,  7.86595013e-01,  4.44828851e-02, -8.46061920e-01,
        -1.75772269e+00,  3.19034542e-01,  4.81801337e-01, -5.02154833e-01,
        -7.00213023e-01, -4.49279353e-01, -6.39765155e-02,  3.30600966e-01,
          7.02834290e-01], 'Steps': 14140, 'MSE': 0.0006038654565721106, 'degree': 8}

  landsf_result =  {'Weights': [ 0.00585349,  0.61867204,  2.17337952, -2.83073102, -2.37963248,
          0.11841184,  2.63007589], 'Steps': 24390, 'MSE': 0.0014746153594685303, 'degree': 6}

  total_finished_area_result = {'Weights': [ 0.00838731,  1.07213719, -0.9134273 ,  0.59034817,  1.33027531,
          1.02969104,  0.17254856, -0.90578411, -2.02932659], 'Steps': 11693, 'MSE': 0.0007063312674572078, 'degree': 8}


  testing(total_finished_area_result)
  validation(total_finished_area_result)