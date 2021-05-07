# Spot-checking is a way of discovering which algorithms 
# perform well on your machine learning problem.

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# ----------------------- Linear Algorithms -------------------------

# Linear regression assumes that the input variables have a Gaussian 
# distribution. It is also assumed that input variables are relevant 
# to the output variable and that they are not highly correlated with 
# each other (a problem called collinearity).
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
# The actual MSE is simply the positive version of the number you're getting.


# Ridge regression is an extension of linear regression where the loss function 
# is modified to minimize the complexity of the model measured as the sum squared 
# value of the coefficient values (also called the l2-norm). Basically a loss 
# is added to the coefficients to minimise the standard error.
model = Ridge()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())


# The Least Absolute Shrinkage and Selection Operator (or LASSO for short) is a 
# modification of linear regression, like ridge regression, where the loss function 
# is modified to minimize the complexity of the model measured as the sum absolute 
# value of the coefficient values (also called the l1-norm).
model = Lasso()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())


# ElasticNet is a form of regularization regression that combines the properties 
# of both Ridge Regression and LASSO regression. It seeks to minimize the complexity 
# of the regression model (magnitude and number of regression coefficients) by 
# penalizing the model using both the l2-norm (sum squared coefficient values) and 
# the l1-norm (sum absolute coefficient values).
model = ElasticNet()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())



# ----------------------- Non Linear Algorithms -------------------------

# K-Nearest Neighbors (or KNN) locates the K most similar instances in the training 
# dataset for a new data instance. From the K neighbors, a mean or median output variable 
# is taken as the prediction. Of note is the distance metric used (the metric argument). 
# The Minkowski distance is used by default, which is a generalization of both the Euclidean 
# distance (used when all inputs have the same scale) and Manhattan distance (for when the 
# scales of the input variables differ).
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())


# Decision trees or the Classification and Regression Trees (CART as they are known) 
# use the training data to select the best points to split the data in order to minimize 
# a cost metric. The default cost metric for regression decision trees is the mean squared 
# error, specified in the criterion parameter.
model = DecisionTreeRegressor()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
