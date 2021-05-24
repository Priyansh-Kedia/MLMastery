from nltk.corpus import stopwords

import string
from os import listdir
from collections import Counter

from keras.preprocessing.text import Tokenizer
from numpy import array

def load_doc(filename):
    file = open(filename,"r")
    text = file.read()
    file.close()
    return text

# Convert doc to token
def clean_doc(doc):
    # Splits the words by space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('','', string.punctuation)
    tokens = [w.translate(table) for w in tokens]

    # remove the tokens which are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    
    # filter out the stop words
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if not w in stop_words]

    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean the doc
    tokens = clean_doc(doc)
    # update the counts
    vocab.update(tokens)

def process_docs(directory, vocab):
    # go to each and every file in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if filename.startswith("cv9"):
            continue
        path = directory + "/" + filename
        add_doc_to_vocab(path, vocab)

def save_list(lines, filename):
    # convert the lines into single blob of text
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# define the vocab using Counter, so that it can store the 
# word and its corresponding frequency
vocab = Counter()
process_docs("txt_sentoken/pos",vocab)
process_docs("txt_sentoken/neg",vocab)

print(len(vocab))
print(vocab.most_common(50))

# keep tokens with a min occurrence
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))

save_list(tokens, "vocab.txt")


def doc_to_line(filename, vocab):
    # load the doc 
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter according to vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def docs_to_lines(directory, vocab):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_lines = docs_to_lines('txt_sentoken/pos', vocab)
negative_lines = docs_to_lines('txt_sentoken/neg', vocab)
# summarize what we have
print(len(positive_lines), len(negative_lines))


# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines


# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the values
docs = negative_lines + positive_lines
tokenizer.fit_on_texts(docs)

# encode the training data set
# mode: one of "binary", "count", "tfidf", "freq".
# if mode == 'count':
#    x[i][j] = c
#    elif mode == 'freq':
#    x[i][j] = c / len(seq)
#    elif mode == 'binary':
#       x[i][j] = 1
#    elif mode == 'tfidf':
#       tf = 1 + np.log(c)
#       idf = np.log(1 + self.document_count /
#                   (1 + self.index_docs.get(j, 0)))
#       x[i][j] = tf * idf
XTrain = tokenizer.texts_to_matrix(docs, mode='freq')
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
print(XTrain.shape)

# load all test reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, False)
negative_lines = process_docs('txt_sentoken/neg', vocab, False)
docs = negative_lines + positive_lines

# encode training data set
Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
print(Xtest.shape)


# Sentiment analysis model

from keras.models import Sequential
from keras.layers import Dense, Dropout

n_words = Xtest.shape[1]
# define network
model = Sequential()
model.add(Dense(50, input_shape=(n_words,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(XTrain, ytrain, epochs=50, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))


def predict_sentiment(review, vocab, tokenizer, model):
    # clean
    tokens = clean_doc(review)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode='freq')

    # prediction
    yhat = model.predict(encoded, verbose=0)
    return round(yhat[0,0])


# test positive text
text = 'Best movie ever!'
print(predict_sentiment(text, vocab, tokenizer, model))
# test negative text
text = 'Not sure how the movie is, what should i be saying?'
print(predict_sentiment(text, vocab, tokenizer, model))