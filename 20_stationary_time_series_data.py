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


# Augmented Dickey-Fuller test
# The Augmented Dickey-Fuller test is a type of statistical test called a unit root test.
# The intuition behind a unit root test is that it determines how strongly a time series 
# is defined by a trend.

# A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis 
# (stationary), otherwise a p-value above the threshold suggests we fail to reject the null 
# hypothesis (non-stationary). p-value > 0.05: Fail to reject the null hypothesis (H0), 
# the data has a unit root and is non-stationary. p-value <= 0.05: Reject the null 
# hypothesis (H0), the data does not have a unit root and is stationary.

from statsmodels.tsa.stattools import adfuller

series = read_csv('daily-total-female-births.csv', header=0, index_col=0, squeeze=True)
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

