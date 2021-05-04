# Load CSV

import csv
import numpy as np

filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)

x = list(reader)
data = np.array(x).astype('float')
print(data.shape)


# Load CSV from URL using Numpy
from numpy import loadtxt
from urllib.request import urlopen

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter=',')
print(dataset.shape)
# Load data from a text file.
# Each row in the text file must have the same number of values.

# Load CSV using Pandas
import pandas as pd

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names)
print(data.shape)

# Load CSV from url using Pandas

data = pd.read_csv(url)
print(data.shape)