from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot 

# generate a dataset
X,Y = make_circles(n_samples=1000, noise=0.1, random_state=1)

# Split into test and train set
trainX, testX = X[:n_train, :], X[n_train:, :]
trainY, testY = Y[:n_train], Y[n_train:]

# define the model
model = Sequential()
model.add(Dense(50, input_dim=2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# compile the model
opt = SGD(lr=0.01,momentum=0.9)
model.compile(loss="binary_crossentropy",optimizer=opt, metrics=['accuracy'])

# defining the learning rate schedule
# monitor: quantity to be monitored.
# factor: factor by which the learning rate will be reduced. new_lr = lr * factor.
# patience: number of epochs with no improvement after which learning rate will be reduced.
# min_delta: threshold for measuring the new optimum, to only focus on significant changes.

rlrp = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_delta=1E-7, verbose=1)

# fit the model
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=300, verbose=0, callbacks=[rlrp])  

# evaluate the model
_, train_acc = model.evaluate(trainX, trainY, verbose=0)
_, test_acc = model.evaluate(testX, testY, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# plot loss learning curves
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

