import pandas as pd
from RegressionMultivariate import *




data = pd.read_csv('Data/Demo/demo_train.csv')

X = data['X']
Y = data['Y']



results = regression(X, Y, limit = 8)

best_result = lowest_mse(results)
print(best_result)

graph(X, Y, best_result['Weights'])

