# A numerical variable can be converted to an ordinal variable by dividing 
# the range of the numerical variable into bins and assigning values to each 
# bin. For example, a numerical variable between 1 and 10 can be divided into 
# an ordinal variable with 5 labels with an ordinal relationship: 1-2, 3-4, 
# 5-6, 7-8, 9-10. This is called discretization.

# Nominal Variable (Categorical). Variable comprises a finite set of discrete 
# values with no relationship between values.
# Ordinal Variable. Variable comprises a finite set of discrete values with a 
# ranked ordering between values.


# Ordinal Encoding
# In ordinal encoding, each unique category value is assigned an integer value.
# For example, “red” is 1, “green” is 2, and “blue” is 3.

from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
# define data
data = asarray([['red'], ['green'], ['blue']])
print(data)
# define ordinal encoding
encoder = OrdinalEncoder()
# transform data
result = encoder.fit_transform(data)
print(result)


# One hot encoding

from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
# define data
data = asarray([['red'], ['green'], ['blue']])
print(data)
# define one hot encoding
encoder = OneHotEncoder(sparse=False)
# transform data
onehot = encoder.fit_transform(data)
print(onehot)


# Dummy variable encoding
# The one-hot encoding creates one binary variable for each category.
# The problem is that this representation includes redundancy. For 
# example, if we know that [1, 0, 0] represents “blue” and [0, 1, 0] 
# represents “green” we don’t need another binary variable to represent 
# “red“, instead we could use 0 values for both “blue” and “green” alone, e.g. [0, 0].
# This is called a dummy variable encoding, and always represents C categories 
# with C-1 binary variables.

from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
# define data
data = asarray([['red'], ['green'], ['blue']])
print(data)
# define one hot encoding
encoder = OneHotEncoder(drop='first', sparse=False)
# transform data
onehot = encoder.fit_transform(data)
print(onehot)


# Breast cancer dataset

from pandas import read_csv
# define the location of the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
# load the dataset
dataset = read_csv(url, header=None)
# retrieve the array of data
data = dataset.values
# separate into input and output columns
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
# summarize
print('Input', X.shape)
print('Output', y.shape)