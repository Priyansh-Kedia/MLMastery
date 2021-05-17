from keras.datasets import mnist
from matplotlib import pyplot as plt

# load the mnist dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data() 

# Plotting the first four images as grayscale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

# baseline model with a single hidden layer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

# load the mnist dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data() 

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype("float32")
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype("float32")

# rescaling the features, as we know the max value of pixel can be 255
X_train = X_train / 255
X_test = X_test / 255

# Now we will use one hot encoding to transform the class integers to a binary matrix
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]