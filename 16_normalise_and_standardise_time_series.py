# A value is normalise as follows
# y = (x - min) / (max - min)

# ----------------------------------------- Normalisation ---------------------------------------

from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
print(series.head())
print(series.shape)

values = series.values
print(values)
values = values.reshape((len(values), 1))

# train the normalization
scaler = MinMaxScaler(feature_range=(0,1))
scaler = scaler.fit(values)
# fit computes the minimum and maximum to be used for later scaling.

print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

# normalise dataset
normalised = scaler.transform(values)
print(normalised)

inversed = scaler.inverse_transform(normalised)
print(inversed)


# ------------------------------------------ Standardisation ------------------------------------------------------

# Standardizing a dataset involves rescaling the distribution of values 
# so that the mean of observed values is 0 and the standard deviation is 1.
# Standardization assumes that your observations fit a Gaussian distribution 
# (bell curve) with a well behaved mean and standard deviation.

# y = (x - mean) / standard_deviation
# mean = sum(x) / count(x)
# standard_deviation = sqrt( sum( (x - mean)^2 ) / count(x))

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import sqrt

series.hist()
plt.show()

values = series.values
values = values.reshape((len(values), 1))
# train the standardization
scaler = StandardScaler()
scaler = scaler.fit(values)
print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))

standardised = scaler.transform(values)
for i in range(5):
	print(standardised[i])
# inverse transform and print the first 5 rows
inversed = scaler.inverse_transform(standardised)
for i in range(5):
	print(inversed[i])