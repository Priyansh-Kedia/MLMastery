import numpy as np
from numpy.linalg import norm

a = np.array([1,2,3])
print(a)
l1 = norm(a,1)
print(l1)

l2 = norm(a,2)
print(l2)

maxnorm = norm(a, np.inf)
print(maxnorm)