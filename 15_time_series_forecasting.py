from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

series = read_csv("daily-births.csv",header=0, parse_dates=[0], index_col=0, squeeze=True)
# header=0: We must specify the header information at row 0.
# parse_dates=[0]: We give the function a hint that data in 
# the first column contains dates that need to be parsed. This 
# argument takes a list, so we provide it a list of one element, 
# which is the index of the first column.
# index_col=0: We hint that the first column contains the index 
# information for the time series.
# squeeze=True: We hint that we only have one data column and that 
# we are interested in a Series and not a DataFrame.

print(type(series))
print(series.head())

print(series.size)
print(series.describe())
print(series['1959-01'])

series.plot()
plt.show()