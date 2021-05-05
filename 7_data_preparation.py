import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names = names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)
# Transform features by scaling each feature to a given range.
np.set_printoptions(precision = 3)
print(rescaledX[0:5,:])


scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# Standardize features by removing the mean and scaling to unit variance
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])

scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# Normalize samples individually to unit norm. (row-wise)
np.set_printoptions(precision=3)
print(normalizedX[0:5,:])

binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
# Binarize data (set feature values to 0 or 1) according to a threshold.
# Values greater than the threshold map to 1, while values less than or 
# equal to the threshold map to 0.
np.set_printoptions(precision=3)
print(binaryX[0:5,:])
