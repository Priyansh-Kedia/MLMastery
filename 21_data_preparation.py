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
