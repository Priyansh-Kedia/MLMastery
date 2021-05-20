from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


X, Y = make_classification(random_state=1)

models = [('knn', KNeighborsClassifier()), ('tree', DecisionTreeClassifier())]

# cv: Determines the cross-validation splitting strategy used in 
# cross_val_predict to train final_estimator. Possible inputs for cv are:
# Defaults to 5
model = StackingClassifier(models, final_estimator=LogisticRegression(), cv=3)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))