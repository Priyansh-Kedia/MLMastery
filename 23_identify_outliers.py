# An outlier is an observation that is unlike the other observations.
# It is rare, or distinct, or does not fit in some way.

from numpy.random import seed, randn
from numpy import mean, std

# seed the random number generator
seed(1)
# generate univariate observations
data = 5 * randn(10000) + 50

# summarize
print('mean=%.3f stdv=%.3f' % (mean(data), std(data)))

# Standard deviation method

data_mean, data_std = mean(data), std(data)
# identify outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
# identify outliers
outliers = [x for x in data if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))
# remove outliers
outliers_removed = [x for x in data if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))


# Interquartile Range Method
# The IQR is calculated as the difference between the 75th and the 25th 
# percentiles of the data and defines the box in a box and whisker plot.

#The IQR can be used to identify outliers by defining limits on the sample 
# values that are a factor k of the IQR below the 25th percentile or above 
# the 75th percentile. The common value for the factor k is the value 1.5. 
# A factor k of 3 or more can be used to identify values that are extreme 
# outliers or “far outs” when described in the context of box and whisker plots.

from numpy import percentile

# seed the random number generator
seed(1)
# generate univariate observations
data = 5 * randn(10000) + 50
# calculate interquartile range
q25, q75 = percentile(data, 25), percentile(data, 75)
iqr = q75 - q25
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
# calculate the outlier cutoff
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off
# identify outliers
outliers = [x for x in data if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))
# remove outliers
outliers_removed = [x for x in data if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))



# Automatic outlier detection

# The local outlier factor, or LOF for short, is a technique that attempts to harness 
# the idea of nearest neighbors for outlier detection. Each example is assigned a scoring 
# of how isolated or how likely it is to be outliers based on the size of its local neighborhood. 
# Those examples with the largest score are more likely to be outliers.

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor

# load the dataset
df = read_csv('boston-housing-dataset.csv', header=None)

# retrieve the array
data = df.values

# split into input and output elements
X, y = data[:, :-1], data[:, -1]

# summarize the shape of the dataset
print(X.shape, y.shape)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the train and test sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
print('MAE: %.3f' % mae)

lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)