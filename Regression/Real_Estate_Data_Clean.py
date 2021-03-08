import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Real_Estate_Sales_730_Days.csv')

data = data[['LandSF','TotalFinishedArea','TotalAppraisedValue']]

  

data = data[data > 0]
data = data[data['TotalAppraisedValue'] < 5000000]
data = data[data['LandSF'] < 350000]



data.to_csv('real_estate.csv', index=False)  

# plt.scatter(data['TotalFinishedArea'],data['TotalAppraisedValue'])
# plt.scatter(data['LandSF'], data['TotalAppraisedValue'])
# plt.show()


