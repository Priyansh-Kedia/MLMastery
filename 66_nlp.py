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

# A bag-of-words is a representation of text that describes the occurrence 
# of words within a document.
# A vocabulary is chosen, where perhaps some infrequently used words are discarded. 
# A given document of text is then represented using a vector with one position for 
# each word in the vocabulary and a score for each known word that appears (or not) 
# in the document.
# It is called a “bag” of words, because any information about the order or structure 
# of words in the document is discarded. The model is only concerned with whether 
# known words occur in the document, not where in the document.

from sklearn.feature_extraction.text import TfidfVectorizer

# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())