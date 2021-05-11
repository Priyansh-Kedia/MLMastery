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


# Feature selection is the process of reducing the number of input 
# variables when developing a predictive model.Recursive Feature 
# Elimination, or RFE for short, is a popular feature selection algorithm.



from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

# Generate a random n-class classification problem.
X,Y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# n_sample = Number of samples
# n_features = Number of features
# n_informative = The number of informative features
# n_redundant = The number of redundant features. These features are generated 
# as random linear combinations of the informative features.

# Define RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
# estimator = A supervised learning estimator with a fit method that provides 
# information about feature importance
# n_features_to_select = The number of features to select

# fit RFE
rfe.fit(X, Y)

for i in range(X.shape[1]):
	print('Column: %d, Selected=%s, Rank: %d' % (i, rfe.support_[i], rfe.ranking_[i]))



# Normalisation = Each of the input variables in scaled to be in between 0-1

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

X, Y = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, random_state=1)

# summarize data before the transform
print(X[:3, :])

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# summarize data after the transform
print(X_norm[:3, :])



# One-Hot Encoding, transformation of categorical data into numeric
# data, as the machine learning algorithms require numerical input.

# Each label for a categorical variable can be mapped to a unique 
# integer, called an ordinal encoding. Then, a one-hot encoding can 
# be applied to the ordinal representation. This is where one new 
# binary variable is added to the dataset for each unique integer 
# value in the variable, and the original categorical variable is 
# removed from the dataset.

# For example, imagine we have a “color” variable with three categories 
# (‘red‘, ‘green‘, and ‘blue‘). In this case, three binary variables are 
# needed. A “1” value is placed in the binary variable for the color and 
# “0” values for the other colors.

from pandas import read_csv
from sklearn.preprocessing import OneHotEncoder

dataset = read_csv('breast-cancer.csv', header=None)

data = dataset.values

# separate into input and output columns
X = data[:, :-1].astype(str)
Y = data[:, -1].astype(str)

print(X[:3, :])

# Defining the encoder 
encoder = OneHotEncoder(sparse=False)
# sparse = Will return sparse matrix if set True else will return an array.

# fit and apply the encoder to the input values
X_oe = encoder.fit_transform(X)

# summarize the transformed data
print(X_oe[:3, :])



# Discretization is the process through which we can transform continuous 
# variables, models or functions into a discrete form. We do this by creating 
# a set of contiguous intervals (or bins) that go across the range of our 
# desired variable/model/function.

from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, random_state=1)

# summarize data before the transform
print(X[:3, :])

# define the transform
trans = KBinsDiscretizer(n_bins=10, encode="ordinal",strategy="uniform")

# transform the data
X_discrete = trans.fit_transform(X)
# summarize data after the transform
print(X_discrete[:3, :])



# Dimensionality reduction with PCA

# The number of input variables or features for a dataset is referred to as its dimensionality.

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=7, random_state=1)
# summarize data before the transform
print(X[:3, :])
# define the transform
trans = PCA(n_components=3)
# n_components tells the number of components to keep after the tranformation

# transform the data
X_dim = trans.fit_transform(X)
# summarize data after the transform
print(X_dim[:3, :])