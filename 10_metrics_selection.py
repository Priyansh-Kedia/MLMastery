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



