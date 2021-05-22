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