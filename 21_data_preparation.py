# Some important steps in data preparation are:

# Data Cleaning: Identifying and correcting mistakes or errors in the data.
# Feature Selection: Identifying those input variables that are most relevant to the task.
# Data Transforms: Changing the scale or distribution of variables.
# Feature Engineering: Deriving new variables from available data.
# Dimensionality Reduction: Creating compact projections of the data.


# Handling missing data
# Real-world data often has missing values, for a number of reasons.
# Filling missing values with data is called data imputation and a 
# popular approach for data imputation is to calculate a statistical 
# value for each column (such as a mean) and replace all missing values 
# for that column with the statistic.

from numpy import isnan, nan
from pandas import read_csv
from sklearn import impute
from sklearn.impute import SimpleImputer

dataframe = read_csv('horse-colic.csv',header=None, na_values="?")

# Splitting into input and output elements
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]

# print total missing
print('Missing: %d' % sum(isnan(X).flatten()))

# defining the imputer
imputer = SimpleImputer(strategy="mean")
imputer.fit(X)

Xtrans = imputer.transform(X)

# print total missing
print('Missing: %d' % sum(isnan(Xtrans).flatten()))