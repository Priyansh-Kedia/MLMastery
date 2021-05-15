import numpy
from keras.models import Sequential
from keras.layers import Dense
# Load the dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]
# Define and Compile

# Sequential groups a linear stack of layers
model = Sequential()

# Dense is a densely-connected Neural network layer
model.add(Dense(12, input_dim=8, activation='relu'))

# activation: Activation function to use. If you don't specify 
# anything, no activation is applied (ie. "linear" activation: a(x) = x).
# first parameter is units: Positive integer, dimensionality 
# of the output space.

model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
# Fit the model
# epochs is the number of iterations on the dataset
# batch_size is the number of samples per batch of computation
model.fit(X, Y, epochs=150, batch_size=10)
# Evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# This code should be run using google collab, so as to 
# avoid version problems