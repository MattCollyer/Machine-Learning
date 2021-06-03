import pandas as pd
import numpy as np




def shuffle_divide_export(df, training_percent = 0.8, validation_percent = 0.1, shuffle = True, path = 'Data/'):
  if(shuffle):
    df = df.sample(frac=1)
  training, validation, test = np.split(df, [int(training_percent*len(df)), int((1-validation_percent)*len(df))])
  training.to_csv(path+'training.csv', index = False)
  validation.to_csv(path+'validation.csv', index = False)
  test.to_csv(path+'test.csv', index = False)



news = pd.read_csv('Data/newsCorpora.csv', sep='\t', names = ["Numeric ID", "Title", "URL", "Publisher", "Category", "Story Id", "URL hostname", "Timestamp"], header = None)
news = news.dropna()

publishers = ['Reuters', 'Huffington Post', 'Businessweek', 'Daily Mail']
news = news[~news.Publisher.isin(publishers)]
news = news[['Title','Category']]

shuffle_divide_export(news)




