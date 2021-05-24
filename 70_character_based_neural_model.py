
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