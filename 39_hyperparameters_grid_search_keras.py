# How to use keras models in scikit learn
# Keras models can be used in scikit-learn by wrapping them with the 
# KerasClassifier or KerasRegressor class.


# Grid search is a model hyperparameter optimization technique.
# In scikit-learn this technique is provided in the GridSearchCV class.

# When constructing this class you must provide a dictionary of 
# hyperparameters to evaluate in the param_grid argument. This is a map 
# of the model parameter name and an array of values to try.

# By default, the grid search will only use one thread. By setting the n_jobs 
# argument in the GridSearchCV constructor to -1, the process will use all cores 
# on your machine. Depending on your Keras backend, this may interfere with the 
# main neural network training process.

# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
# n_jobs = -1 may not run as it uses all cores of the machine

grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



