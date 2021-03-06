import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# We will see an example of 1D Convolution

from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv1D

# define input data
data = asarray([0, 0, 0, 1, 1, 0, 0, 0])
data = data.reshape(1, 8, 1)

# create model
model = Sequential()
model.add(Conv1D(1, 3, input_shape=(8, 1)))

# define a vertical line detector
weights = [asarray([[[0]],[[1]],[[0]]]), asarray([0.0])]

# store the weights in the model
model.set_weights(weights)

# confirm they were stored
print(model.get_weights())

# apply filter to input data
yhat = model.predict(data)
print(yhat)


from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D

# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
data = data.reshape(1, 8, 8, 1)

# create model
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))

# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]

# store the weights in the model
model.set_weights(weights)

# confirm they were stored
print(model.get_weights())

# apply filter to input data
yhat = model.predict(data)
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])