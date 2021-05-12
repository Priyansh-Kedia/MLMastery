# Machine learning algorithms like Linear Regression and Gaussian 
# Naive Bayes assume the numerical variables have a Gaussian probability distribution.

# Your data may not have a Gaussian distribution and instead may have 
# a Gaussian-like distribution (e.g. nearly Gaussian but with outliers 
# or a skew) or a totally different distribution (e.g. exponential).

# As such, you may be able to achieve better performance on a wide range 
# of machine learning algorithms by transforming input and/or output 
# variables to have a Gaussian or more-Gaussian distribution. Power 
# transforms like the Box-Cox transform and the Yeo-Johnson transform 
# provide an automatic way of performing these transforms on your data


# Power Transforms
# We can apply a power transform directly by calculating the log or square 
# root of the variable, although this may or may not be the best power 
# transform for a given variable.

# There are two popular approaches for such automatic power transforms; they are:
# Box-Cox Transform
# Yeo-Johnson Transform
# A hyperparameter, often referred to as lambda  is used to control the nature 
# of the transform.

# lambda = -1. is a reciprocal transform.
# lambda = -0.5 is a reciprocal square root transform.
# lambda = 0.0 is a log transform.
# lambda = 0.5 is a square root transform.
# lambda = 1.0 is no transform.

# We will be using PowerTransformer class
# The class takes an argument named “method” that can be set to ‘yeo-johnson‘ 
# or ‘box-cox‘ for the preferred method. It will also standardize the data 
# automatically after the transform, meaning each variable will have a zero 
# mean and unit variance. This can be turned off by setting the “standardize” 
# argument to False.

from numpy import exp
from numpy.random import randn
from sklearn.preprocessing import PowerTransformer
from matplotlib import pyplot
# generate gaussian data sample
data = randn(1000)
# add a skew to the data distribution
data = exp(data)
# histogram of the raw data with a skew
pyplot.hist(data, bins=25)
pyplot.show()
# reshape data to have rows and columns
data = data.reshape((len(data),1))
# power transform the raw data
power = PowerTransformer(method='yeo-johnson', standardize=True)
data_trans = power.fit_transform(data)
# histogram of the transformed data
pyplot.hist(data_trans, bins=25)
pyplot.show()





