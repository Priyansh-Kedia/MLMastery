import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report



# ----------------------------------- Classification Metrics ------------------------------------------------------

# Classification accuracy is the number of correct predictions made as a ratio of all predictions made.
# It is really only suitable when there are an equal number of observations in each class 
# (which is rarely the case) and that all predictions and prediction errors are equally 
# important, which is often not the case.
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# Log loss (Logistic Loss) is a performance metric for evaluating the predictions 
# of probabilities of membership to a given class. The scalar probability between 
# 0 and 1 can be seen as a measure of confidence for a prediction by an algorithm. 
# Predictions that are correct or incorrect are rewarded or punished proportionally 
# to the confidence of the prediction.

scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# Area under ROC curve(or ROC AUC for short) is a performance metric for binary 
# classification problems. The AUC represents a model’s ability to discriminate 
# between positive and negative classes. An area of 1.0 represents a model that 
# made all predictions perfectly. An area of 0.5 represents a model as good as random.
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))


# Confusion Matrix is a handy presentation of the accuracy of a model with two or 
# more classes. The table presents predictions on the x-axis and accuracy outcomes 
# on the y-axis. The cells of the table are the number of predictions made by a 
# machine learning algorithm.
test_size = 0.33
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=7)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)


# Classification Report 
test_size = 0.33
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=7)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)


# ----------------------------------- Regression Metrics ------------------------------------------------------

# Mean absolute error is the average of the absolute differences between predictions 
# and actual values. It gives an idea of how wrong the predictions were.
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))


# Mean Squared Error is much like the mean absolute error in that it provides 
# a gross idea of the magnitude of error. Taking the square root of the mean 
# squared error converts the units back to the original units of the output 
# variable and can be meaningful for description and presentation. This is 
# called the Root Mean Squared Error (or RMSE).
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)" % (results.mean(), results.std()))


# R^2 Metric provides an indication of the goodness of fit of a set of 
# predictions to the actual values. In statistical literature, this 
# measure is called the coefficient of determination. This is a value 
# between 0 and 1 for no-fit and perfect fit respectively.
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'r2'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))

