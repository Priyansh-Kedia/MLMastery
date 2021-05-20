# Rather than considering all features when choosing a split point, 
# random forest limits the features to a random subset of features, 
# such as 3 if there were 10 features.

from numpy import mean, std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

X,Y = make_classification(random_state=1)

# number of randomly selected features to consider at each split point 
# via the “max_features” argument, which is set to the square root of 
# the number of features in your dataset by default.
model = RandomForestClassifier(n_estimators=50)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=1)

# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))