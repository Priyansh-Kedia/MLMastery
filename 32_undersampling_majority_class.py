# A simple approach to using standard machine learning algorithms on an 
# imbalanced dataset is to change the training dataset to have a more 
# balanced class distribution.

# This can be achieved by deleting examples from the majority class, 
# referred to as “undersampling.” A possible downside is that examples 
# from the majority class that are helpful during modeling may be deleted.

# We use the function from imbalanced-learn library.



from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler

X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99, 0.01], flip_y=0)
# summarize class distribution
print(Counter(y))
# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy=0.5)
# When sampling_strategy is 0.5, then the minority class will have half the number of
# values of the majority class after tranformation

# fit and apply the transform
X_under, y_under = undersample.fit_resample(X, y)
# summarize class distribution
print(Counter(y_under))