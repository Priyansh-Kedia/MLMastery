# Imbalanced Classification: A classification predictive modeling problem 
# where the distribution of examples across the classes is not equal.

from numpy import where
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

# cluster_std = The standard deviation of the clusters.
# centers = Specifies the number of classes to generate, it is 
# called centers because a class cluster tend to gather close to a center.
X,Y = make_blobs(n_samples=1000, centers=2,n_features=2, random_state=1, cluster_std=3)

for class_value in range(2):
    # get row indexes for samples with this class
    # The numpy.where() function returns the indices of elements in an input array 
    # where the given condition is satisfied.
    row_ix = where(Y == class_value)

    plt.scatter(X[row_ix, 0],X[row_ix,1])

plt.show()

# make_blobs always returns an equal class distribution

