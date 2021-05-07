# Spot-checking is a way of discovering which algorithms 
# perform well on your machine learning problem.

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

# ----------------------- Linear Algorithms -------------------------

# Logistic regression assumes a Gaussian distribution 
# for the numeric input variables and can model binary classification problems.
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())



