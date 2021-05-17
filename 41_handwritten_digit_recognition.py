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

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    # dense layer out is calculated as follows
    # output = activation(dot(input, kernel) + bias)
    # kernel is a weights matrix created by the layer


    # first param is the dimensions of the output from this layer
    # second param is the input dimensions of this layer, which is 28*28 = 784 in this case
    # kernel_regularizer: Regularizer function applied to the kernel weights matrix.
    # activation: Activation function to use. If you don't specify anything, no activation 
    # is applied (ie. "linear" activation: a(x) = x).
    # Read about the different activations here https://keras.io/api/layers/activations/
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer="normal", activation="relu"))
    model.add(Dense(num_classes, kernel_initializer="normal", activation="softmax"))

    # compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

