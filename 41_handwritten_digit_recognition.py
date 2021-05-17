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
    # The loss function to be used. To optimise a solution, loss should be least
    # Optimizers are Classes or methods used to change the attributes of your 
    # machine/deep learning model such as weights and learning rate in order to reduce the losses.
    # Read about different kinds of optimizers here https://keras.io/api/optimizers/

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# build the model
model = baseline_model()
# fit the model
# batch_size: Integer or None. Number of samples per gradient update. If unspecified, 
# batch_size will default to 32
# epochs: Integer. Number of epochs to train the model. An epoch is an iteration over 
# the entire x and y data provided.
# verbose: 'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one 
# line per epoch. 'auto' defaults to 1 for most cases
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
# final evaluation of the model

# verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


# Now we create a CNN to reduce the error
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

# In Keras, the layers used for two-dimensional convolutions expect pixel values with the 
# dimensions [pixels][width][height][channels].
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Normalizing the values
X_train = X_train / 255
X_test = X_test / 255
# one hot encoding of outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

# The first hidden layer is a convolutional layer called a Convolution2D. The layer has 32 
# feature maps, which with the size of 5×5 and a rectifier activation function. This is the input 
# layer, expecting images with the structure outline above [pixels][width][height].
# Next we define a pooling layer that takes the max called MaxPooling2D. It is configured with a pool size of 2×2.
# The next layer is a regularization layer using dropout called Dropout. It is configured to randomly exclude 
# 20% of neurons in the layer in order to reduce overfitting.
# Next is a layer that converts the 2D matrix data to a vector called Flatten. It allows the output to 
# be processed by standard fully connected layers.
# Next a fully connected layer with 128 neurons and rectifier activation function.
# Finally, the output layer has 10 neurons for the 10 classes and a softmax activation function to output 
# probability-like predictions for each class.

def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))