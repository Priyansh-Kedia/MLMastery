# The observations in a stationary time series are not dependent on time.
# Time series are stationary if they do not have trend or seasonal effects.

from pandas import read_csv
from matplotlib import pyplot as plt

series1 = read_csv('daily-total-female-births.csv', header=0, index_col=0)
series1.plot()
plt.show()

series2 = read_csv('international-airlines-passengers.csv', header=0, index_col=0)
series2.plot()
plt.show()

# A quick and dirty way of checking in data in stationary or not, 
# you can split your time series into two (or more) partitions and 
# compare the mean and variance of each group. If they differ and the 
# difference is statistically significant, the time series is likely non-stationary.

# Summary statistics for stationary data
X = series1.values
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var() # variance
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))


# Summary statistics for non-stationary data
X = series2.values
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))