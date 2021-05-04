from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# In this case, random state basically specifies that your data will be split in a specific order always
# For example, the order you will get in random_state=0 remain same. After that if you execute
# random_state=5 and again come back to random_state=0 you'll get the same order.

# Fit the model on 67%
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)

# save the model to disk
filename = 'finalized_model.pickle'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

# WE will be able to see that accuracy score and model.score
# are exactly the same.

predictions = loaded_model.predict(X_test)
print(accuracy_score(Y_test, predictions))
