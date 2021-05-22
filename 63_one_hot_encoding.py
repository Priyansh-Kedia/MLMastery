# Manual One Hot Encoding
from numpy import argmax

data = "hello world"
print(data)

# define universe of possible input values
alphabet = 'abcdefghijklmnopqrstuvwxyz '

# define a mapping of chars to integers
char_to_int = dict((c,i) for i, c in enumerate(alphabet))
int_to_char = dict((i,c) for i, c in enumerate(alphabet))
integer_encoded = [char_to_int[char] for char in data]
print(integer_encoded)

# integer encode the input data
onehot_encoded = list()
for value in integer_encoded:
    letter = [0 for _ in range(len(alphabet))]
    letter[value] = 1
    onehot_encoded.append(letter)

print(onehot_encoded)

# argmax: Returns the indices of the maximum values along an axis.
inverted = int_to_char[argmax(onehot_encoded[0])]
print(inverted)


# One hot encoding using scikit-learn 
from numpy import array, argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)

# encode values
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)


# One hot encoding with keras

from numpy import array, argmax
from keras.utils import to_categorical

# define example
data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
data = array(data)
print(data)

# one hot encode
encoded = to_categorical(data)
print(encoded)

# invert encoding
inverted = argmax(encoded[0])
print(inverted)