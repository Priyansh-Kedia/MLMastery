# Gradient boosting is a framework for boosting ensemble algorithms 
# and an extension to AdaBoost.

# It re-frames boosting as an additive model under a statistical framework 
# and allows for the use of arbitrary loss functions to make it more flexible 
# and loss penalties (shrinkage) to reduce overfitting.

# Gradient boosting also introduces ideas of bagging to the ensemble members, 
# such as sampling of the training dataset rows and columns, referred to as 
# stochastic gradient boosting.

# It is a very successful ensemble technique for structured or tabular data, 
# although it can be slow to fit a model given that models are added sequentially. 
# More efficient implementations have been developed, such as the popular extreme 
# gradient boosting (XGBoost) and light gradient boosting machines (LightGBM).

from numpy import mean, std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

X,Y = make_classification(random_state=1)

model = GradientBoostingClassifier(n_estimators=50)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, X, Y, cv=cv, scoring='accuracy', n_jobs=1)

# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

