
# load doc 
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename,'w')
    file.write(data)
    file.close()

raw_text = load_doc("rhyme.txt")
print(raw_text)

tokens = raw_text.split()
raw_text = ' '.join(tokens)
print(raw_text)

# organise into sequences of characters
length = 10
sequences = list()
for i in range(length, len(raw_text)):
    # select sequence of tokens
    seq = raw_text[i-length:i+1]
    sequences.append(seq)

print("Sequences ",sequences)
print("Sequences length ", len(sequences))

# save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# load data
filename = "char_sequences.txt"
raw_text = load_doc(filename)
lines = raw_text.split('\n')

# We encode the sequences to integer values
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
    encoded_seq = [mapping[char] for char in line]
    sequences.append(encoded_seq)

vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)
print("Sequence", sequences)

# Split into input and output sequence of characters
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]

# Ended here, due to lack of understanding, WTF!