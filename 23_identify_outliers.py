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
