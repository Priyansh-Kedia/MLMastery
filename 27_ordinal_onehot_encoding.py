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

