# Data cleaning

from os import read
from pandas import read_csv
from numpy import unique
from urllib.request import urlopen
from numpy import loadtxt

# define the location of the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/oil-spill.csv'
# load the dataset
data = loadtxt(urlopen(path), delimiter=',')

# summarize the number of unique values in each column
for i in range(data.shape[1]):
	print(i, len(unique(data[:, i])))
    # We can see that, column 22 has only one unique value


# We can do the same using pandas

data = read_csv('oil-spill.csv', header=None)
print(data.nunique())

# Now we delete the column(s) which have only one unique value
counts = data.nunique()
columns_to_delete = [i for i,v in enumerate(counts) if v == 1]
print(columns_to_delete)

new = data.drop(columns_to_delete, axis=1, inplace=False)
# If True, original `data` is changed, else a copy is made and then changed
# For true, no need to store it in variable `new`, `data` will be changed itself.

print(data.shape)
print(new.shape)


# We can remove the columns which have few unique values

columns_to_delete = [i for i,v in enumerate(counts) if (float(v)/data.shape[0]*100) < 1]
print(columns_to_delete)

new = data.drop(columns_to_delete, axis=1, inplace=False)
print(new.shape)


# Remove columns with low variance

from sklearn.feature_selection import VarianceThreshold

df = read_csv("oil-spill.csv", header=None)
# split data into inputs and outputs
data = df.values
X = data[:, :-1]
y = data[:, -1]
print(X.shape, y.shape)
# define the transform
transform = VarianceThreshold()
# The default value of parameter in `VarianceThreshold` is `threshold` is 0

# transform the input data
X_sel = transform.fit_transform(X)
print(X_sel.shape)
# As the threshold is 0, only column 22 is removed


# Removal of duplicate rows

from pandas import read_csv

data = read_csv('iris.csv',header=0, index_col=0)
print(data)