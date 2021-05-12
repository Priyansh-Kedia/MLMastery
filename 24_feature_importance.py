# Feature importance refers to a class of techniques for assigning scores 
# to input features to a predictive model that indicates the relative 
# importance of each feature when making a prediction.


# We will create test dataset with 5 important and 5 unimportant features
# Dataset would be created both for classification as well as regression

from sklearn.datasets import make_classification

# Create a classification dataset
seed = 32
X, Y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=seed)

# Create a regresion dataset
from sklearn.datasets import make_regression

X_reg, Y_reg = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=seed)



