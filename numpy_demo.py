import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mylist = [1,2,3]
myarray = np.array(mylist)
print(mylist)
print(myarray)
print(myarray.shape)

mylist = [[1, 2, 3], [3, 4, 5]]
myarray = np.array(mylist)
print(myarray)
print(myarray.shape)
print("First row: %s" % myarray[0])
print("Last row: %s" % myarray[-1])
print("Specific row and col: %s" % myarray[0, 2])
print("Whole col: %s" % myarray[:, 2])

myarray1 = np.array([2, 2, 2])
myarray2 = np.array([3, 3, 3])
print("Addition: %s" % (myarray1 + myarray2))
print("Multiplication: %s" % (myarray1 * myarray2))


# matplotlib
myarray = np.array([1,2,3])
plt.plot(myarray)
plt.xlabel("X Axis label")
plt.ylabel("Y Axis label")
plt.show()
# If you provide a single list or array to plot , matplotlib assumes 
# it is a sequence of y values, and automatically generates the x values 
# for you. Since python ranges start with 0, the default x vector has the 
# same length as y but starts with 0. Hence the x data are [0, 1, 2, 3]

x = np.array([1, 2, 3])
y = np.array([2, 4, 6])
plt.scatter(x,y)
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.show()


# Pandas demo
myarray = np.array([1,2,3])
rownames = ['a','b','c']
myseries = pd.Series(myarray, index = rownames)
print(myseries)
print(myseries[0])
print(myseries['a'])
# A series is a one-dimensional array where the rows and columns can be labeled.

