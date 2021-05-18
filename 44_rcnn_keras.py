# Time series prediction problems are a difficult type of predictive 
# modeling problem.

# Unlike regression predictive modeling, time series also adds the 
# complexity of a sequence dependence among the input variables.

# A powerful type of neural network designed to handle sequence 
# dependence is called recurrent neural networks. The Long Short-Term 
# Memory network or LSTM network is a type of recurrent neural network 
# used in deep learning because very large architectures can be 
# successfully trained.

import pandas as pd
import matplotlib.pyplot as plt

# engine: Parser engine to use. The C engine is faster while the 
# python engine is currently more feature-complete.
dataset = pd.read_csv("international-airlines-passengers.csv", usecols=[1],engine="python")

plt.plot(dataset)
plt.show()

# The Long Short-Term Memory network, or LSTM network, is a recurrent 
# neural network that is trained using Backpropagation Through Time and 
# overcomes the vanishing gradient problem.

# Instead of neurons, LSTM networks have memory blocks that are connected through layers.

# A block has components that make it smarter than a classical neuron and a memory for 
# recent sequences. A block contains gates that manage the block’s state and output. A 
# block operates upon an input sequence and each gate within a block uses the sigmoid 
# activation units to control whether they are triggered or not, making the change of 
# state and addition of information flowing through the block conditional.

# There are three types of gates within a unit:

# Forget Gate: conditionally decides what information to throw away from the block.
# Input Gate: conditionally decides which values from the input to update the memory state.
# Output Gate: conditionally decides what to output based on input and the memory of the block.

# LSTM for regression
import numpy 
from matplotlib import pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Fix the random seed 
numpy.random.seed(7)

# import the dataset
dataset = pd.read_csv("international-airlines-passengers.csv", usecols=[1],engine="python")

plt.plot(dataset)
plt.show()

# scale the data as LSTM is sensitive to the scale of the data
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

# split the data, as the data has time factor, we cannot split it randomly
# It has to be split in order
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# creating a new dataset, which will have value for t and t+1
def create_dataset(dataset, look_back=1):
    dataX, dataY = [],[]
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# The LSTM network expects the input data (X) to be provided with 
# a specific array structure in the form of: [samples, time steps, features].
# Currently, our data is in the form: [samples, features] and we are framing 
# the problem as one time step for each sample.
# reshape the data
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create the model
model = Sequential()

# units: Positive integer, dimensionality of the output space.
# By default, the recurrent activation is sigmoid
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
