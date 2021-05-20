from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier

X, Y = make_classification(random_state=1)

# n_estimators specifies the number of trees to create
model = BaggingClassifier(n_estimators=50)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# n_jobs: Number of jobs to run in parallel. When -1, uses all processors

n_scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=1)

# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))