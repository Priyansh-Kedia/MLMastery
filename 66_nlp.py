# Textual data must be cleaned first

# Manual Tokenisation
# Tokenisation: It is the process of turning raw
# text into something which can be used to train
# the model. The words after tokenisation are called tokens

filename = "sample3.txt"
file = open(filename,"rt")
text = file.read()

file.close()

words = text.split()

# convert to lowercase
words = [word.lower() for word in words]
# print(words)

from nltk.tokenize import word_tokenize
import nltk

tokens = word_tokenize(text)
print(tokens)