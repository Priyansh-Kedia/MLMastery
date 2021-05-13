# An alternative to deleting examples from the majority class is to add 
# new examples from the minority class.

# This can be achieved by simply duplicating examples in the minority class, 
# but these examples do not add any new information. Instead, new examples 
# from the minority can be synthesized using existing examples in the training 
# dataset. These new examples will be “close” to existing examples in the 
# feature space, but different in small but random ways.

# The SMOTE algorithm is a popular approach for oversampling the minority class. 
# This technique can be used to reduce the imbalance or to make the class distribution even.

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99, 0.01], flip_y=0)
# summarize class distribution
print(Counter(y))
# define oversample strategy
oversample = SMOTE(sampling_strategy=0.5)
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
# summarize class distribution
print(Counter(y_over))


