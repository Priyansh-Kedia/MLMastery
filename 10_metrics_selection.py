import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

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

