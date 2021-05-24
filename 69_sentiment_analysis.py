from nltk.corpus import stopwords

import string
from os import listdir
from collections import Counter

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