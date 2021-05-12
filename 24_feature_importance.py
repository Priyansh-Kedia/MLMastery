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


# Linear regression feature importance
# We can fit a linear regression model on the data, and then get the coefficient values for the 
# input features. This assumes that the input features had the same scale before training

from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

model = LinearRegression()

model.fit(X_reg, Y_reg)

importance = model.coef_

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# Logistic Regresion feature importance
# The same process of retrieval of coefficients after the model is fit can be applied
# on Logistic Regression. This also assumes that the input features had the same scale before training

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X, Y)
importance = model.coef_[0]
# In case of Logistic Regression, coef_ returns a 2D array, this can be checked by print(model.coef_)

for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
# We will be able to see that there a negative coefficients too, which does not mean, 
# that they are of less importance. A positive coefficient indicates the weight of the 
# features towards 1, and negative coefficient indicates towards 0, in a problem where
# 0 and 1 are classes

