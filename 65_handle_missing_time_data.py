# Remove missing sequence data

from random import random
from numpy import array
from pandas import concat
from pandas import DataFrame

# generate a sequence of random data
def generate_sequence(n_timesteps):
    return [random() for _ in range(n_timesteps)]

# generate data for lstm
def generate_data(n_timesteps):
    # generate the sequence
    sequence = generate_sequence(n_timesteps)
    sequence = array(sequence)

    # create lag
    df = DataFrame(sequence)
    df = concat([df.shift(1),df], axis=1)

    # remove rows with missing values
    df.dropna(inplace=True)
    values = df.values

    X, Y = values, values[:,0]
    return X,Y

n_timesteps = 10

X,Y = generate_data(n_timesteps)

# print sequence
for i in range(len(X)):
	print(X[i], '=>', Y[i])


# Replace missing values

from random import random
from numpy import array
from pandas import concat
from pandas import DataFrame
 
# generate a sequence of random values
def generate_sequence(n_timesteps):
	return [random() for _ in range(n_timesteps)]

# generate data for the lstm
def generate_data(n_timesteps):
	# generate sequence
	sequence = generate_sequence(n_timesteps)
	sequence = array(sequence)
	# create lag
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	# replace missing values with -1
	df.fillna(-1, inplace=True)
	values = df.values
	# specify input and output data
	X, Y = values, values[:, 1]
	return X, Y


# generate sequence
n_timesteps = 10

X, Y = generate_data(n_timesteps)

# print sequence
for i in range(len(X)):
	print(X[i], '=>', Y[i])