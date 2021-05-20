# Rather than full decision trees, AdaBoost uses very simple 
# trees that make a single decision on one input variable before 
# making a prediction. These short trees are referred to as decision 
# stumps.

from numpy import mean, std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier

X,Y = make_classification(random_state=1)

model = AdaBoostClassifier(n_estimators=50)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, X, Y, cv=cv, scoring='accuracy', n_jobs=1)

# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))