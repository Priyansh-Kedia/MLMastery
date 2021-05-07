# Spot-checking is a way of discovering which algorithms 
# perform well on your machine learning problem.

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

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


