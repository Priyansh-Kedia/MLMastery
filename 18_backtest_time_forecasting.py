# We often use methods such as using train-test splits and k-fold 
# cross validation to make sure that accurate predictions are made. 
# But these kind of methods do not work with time series data.

# Evaluation of machine learning models on time series data is called 
# backtesting or hindcasting.

# The methods like train-test splits and k-fold do not directly work
# with time series data, as these methods assume that each observation
# is independent. But this is not the case of time series data, the 
# data is mutually exclusive. So data should be split in the order in 
# which it was observed.

from pandas import read_csv
from matplotlib import pyplot as plt

series = read_csv('monthly-sunspots.csv',index_col=0, header=0)
print(series.head())
series.plot()
plt.show()


# Train-test split, for time series data
X = series.values
train_size = int(len(X) * 0.66)
train, test = X[0:train_size], X[train_size:len(X)]
print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))
plt.plot(train)
plt.plot([None for i in train] + [x for x in test])
plt.show()