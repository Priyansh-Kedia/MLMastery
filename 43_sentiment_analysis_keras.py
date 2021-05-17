import numpy
from keras.datasets import imdb
from matplotlib import pyplot as plt

# load the dataset
(X_train, Y_train), (X_test, Y_test) = imdb.load_data()
X = numpy.concatenate((X_train,X_test), axis=0)
Y = numpy.concatenate((Y_train, Y_test), axis=0)
print("Data")
print(X.shape)

# Summarise the number of classes
print("Classes: ")
print(numpy.unique(Y))

# Summarise number of words
print("Number of words: ")
print(len(numpy.unique(numpy.hstack(X))))

# Summarize review length
print("Review length: ")
result = [len(x) for x in X]
print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))
# plot review length
plt.boxplot(result)
plt.show()


# A recent breakthrough in the field of natural language processing is called word embedding.

# This is a technique where words are encoded as real-valued vectors in a high-dimensional space, 
# where the similarity between words in terms of meaning translates to closeness in the vector space.

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# load the dataset, keeping only the top n words
top_words = 5000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=top_words)
max_words = 500
# pad_sequences makes this sequences of words to be of 500 words (integers, as the words
# are changed to integers according to their occurance in the dataset)
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
model = Sequential()

# Turns positive integers (indexes) into dense vectors of fixed size.
# first param is the input dimension
# second param is the output dimension
# input_length: Length of input sequences, when it is constant. This 
# argument is required if you are going to connect Flatten then Dense 
# layers upstream (without it, the shape of the dense outputs cannot be computed).
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=2,batch_size=128, verbose=2)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# a better cnn model for imdb

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# pad dataset to a maximum review length in words
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))