from pandas import DataFrame, concat
from keras.models import Sequential
from keras.layers import LSTM, Dense

# create sequence
length = 10
sequence = [i / float(length) for i in range(length)]
print(sequence)

# create X, Y pairs
df = DataFrame(sequence)
df = concat([df.shift(1),df], axis=1)
df.dropna(inplace=True)
# convert to LSTM friendly format
values = df.values
X, Y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)

# define network
model = Sequential()

# units: Positive integer, dimensionality of the output space.
# units is the first parameter of LSTM
model.add(LSTM(10, input_shape=(1,1)))
model.add(Dense(1))

# compile the network
model.compile(optimizer='adam',loss='mean_squared_error')

# fit the network
history = model.fit(X,Y, epochs=1000,batch_size=len(X), verbose=2)

# 4. evaluate network
loss = model.evaluate(X, Y, verbose=0)
print(loss)
# 5. make predictions
predictions = model.predict(X, verbose=0)
print(predictions[:, 0])