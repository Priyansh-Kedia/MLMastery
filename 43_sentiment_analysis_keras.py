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

