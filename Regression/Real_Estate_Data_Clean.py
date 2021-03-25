import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def shuffle_divide_export(df, training_percent = 0.8, validation_percent = 0.1, shuffle = True, path = 'Data/Housing/'):
  if(shuffle):
    df = df.sample(frac=1)
  training, validation, test = np.split(df, [int(training_percent*len(df)), int((1-validation_percent)*len(df))])
  training.to_csv(path+'training.csv', index = False)
  validation.to_csv(path+'validation.csv', index = False)
  test.to_csv(path+'test.csv', index = False)




data = pd.read_csv('Data/Housing/Real_Estate_Sales_730_Days.csv')

data = data[['LandSF','TotalFinishedArea','TotalAppraisedValue']]
data = data.dropna()
data = data[data > 0]
data = data[data['TotalAppraisedValue'] < 5000000]
data = data[data['LandSF'] < 350000]


shuffle_divide_export(data)

# plt.scatter(data['TotalFinishedArea'],data['TotalAppraisedValue'])
# plt.scatter(data['LandSF'], data['TotalAppraisedValue'])
# plt.show()


