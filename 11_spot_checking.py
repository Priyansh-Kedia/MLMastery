# Spot-checking is a way of discovering which algorithms 
# perform well on your machine learning problem.

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

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


# Linear Discriminant Analysis or LDA is a statistical technique for binary and 
# multi-class classification. It too assumes a Gaussian distribution for the 
# numerical input variables.
model = LinearDiscriminantAnalysis()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())



# ----------------------- Non Linear Algorithms -------------------------


# K-Nearest Neighbors (or KNN) uses a distance metric to find the K most 
# similar instances in the training data for a new instance and takes the 
# mean outcome of the neighbors as the prediction.
model = KNeighborsClassifier()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# Naive Bayes calculates the probability of each class and the conditional 
# probability of each class given each input value. These probabilities are 
# estimated for new data and multiplied together, assuming that they are 
# all independent (a simple or naive assumption).
model = GaussianNB()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# Classification and Regression Trees (CART or just decision trees) construct 
# a binary tree from the training data. Split points are chosen greedily by 
# evaluating each attribute and each value of each attribute in the training 
# data in order to minimize a cost function (like Gini).
model = DecisionTreeClassifier()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

