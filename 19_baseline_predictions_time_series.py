# A baseline in forecast performance provides a point of comparison.

# To estimate the baseline, often zero rule algorithm is used.
# This algorithm predicts the majority class in the case of classification,
# or the average outcome in the case of regression. This could be used for 
# time series, but does not respect the serial correlation structure in time series datasets.

# There is a similar algorithm, persistance algorithm
# The persistence algorithm uses the value at the previous time step (t-1) 
# to predict the expected outcome at the next time step (t+1).

from pandas import read_csv, datetime, DataFrame, concat
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

# persistence model
def model_persistence(x):
	return x

series = read_csv('shampoo-sales.csv',header=0, index_col=0, parse_dates=[0], squeeze=True, date_parser=parser)
series.plot()
plt.show()

values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1','t+1']
print(dataframe.head())

X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]


# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)

# plot predictions and expected results
plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y])
plt.plot([None for i in train_y] + [x for x in predictions])
plt.show()