import pandas as pd
from Regression import *
import numpy as np



data = pd.read_csv('Data/Housing/training.csv')

# X = data['X']
# Y = data['Y']
data = normalize(data)
X1 = data['LandSF']
X2 = data['TotalFinishedArea']
Y = data['TotalAppraisedValue']

X = np.column_stack((X1, X2))


# results = regression(X, Y, limit = 8)

# best_result = lowest_mse(results)
# print(best_result)

W = [ 1.07878098e-03,  8.11974341e-01, -5.87133491e-01,  6.59019988e-01,
        1.12655324e+00,  7.86595013e-01,  4.44828851e-02, -8.46061920e-01,
       -1.75772269e+00,  3.19034542e-01,  4.81801337e-01, -5.02154833e-01,
       -7.00213023e-01, -4.49279353e-01, -6.39765155e-02,  3.30600966e-01,
        7.02834290e-01]


Y = np.column_stack((Y,Y))
graph(X, Y, W)

