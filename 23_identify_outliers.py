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

