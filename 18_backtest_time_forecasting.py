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


# Multiple test train splits

# training_size = i * n_samples / (n_splits + 1) + n_samples % (n_splits + 1)
# test_size = n_samples / (n_splits + 1)
# Where n_samples is the total number of observations, n_splits is the total 
# number of splits and i is the iterator value
from sklearn.model_selection import TimeSeriesSplit

splits = TimeSeriesSplit(n_splits=3)
plt.figure(1)
index = 1
for train_index, test_index in splits.split(X):
	train = X[train_index]
	test = X[test_index]
	print('Observations: %d' % (len(train) + len(test)))
	print('Training Observations: %d' % (len(train)))
	print('Testing Observations: %d' % (len(test)))
	plt.subplot(310 + index) # Read about what this does
	plt.plot(train)
	plt.plot([None for i in train] + [x for x in test])
	index += 1

plt.show()



# Walk forward validation
# In practice, we very likely will retrain our model as new data becomes available.

n_train = 500
n_records = len(X)
for i in range(n_train, n_records):
	train, test = X[0:i], X[i:i+1]
	print('train=%d, test=%d' % (len(train), len(test)))
# This is not expensive if the modeling method is simple or dataset is small (as in 
# this example), but could be an issue at scale.