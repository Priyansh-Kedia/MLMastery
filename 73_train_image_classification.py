from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D ,Dense, Flatten

(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

# reshape dataset to have a single channel
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

# convert from integers to floats
trainX, testX = trainX.astype('float32'), testX.astype('float32')

# normalize to range 0-1
trainX,testX  = trainX / 255.0, testX / 255.0

# one hot encode target values
trainY, testY = to_categorical(trainY), to_categorical(testY)

# define model
model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu',kernel_initializer='he_uniform', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# fit the model
model.fit(trainX, trainY, epochs=10, batch_size = 32, verbose=2)

# evaluate model
loss, acc = model.evaluate(testX, testY, verbose=0)
print(loss, acc)



# Image Augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from matplotlib import pyplot

image = load_img("bird.jpg")

data = img_to_array(image)

# expand dimensions to one sample
data = expand_dims(data, 0)

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)

it = datagen.flow(data, batch_size=1)

# generate samples and plot
for i in range(9):
     # define subplot
     pyplot.subplot(330 + 1 + i)
     # generate batch of images
     batch = it.next()
     # convert to unsigned integers for viewing
     image = batch[0].astype('uint32')
     # plot raw pixel data
     pyplot.imshow(image)
# show the figure
pyplot.show()