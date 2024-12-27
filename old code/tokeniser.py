import re
from nietzsche import *
from constants import *
from tokenizers import ByteLevelBPETokenizer

# tokenizer = ByteLevelBPETokenizer.from_file("vocab-vocab.json", "vocab-merges.txt")

''' Old tokenizer just didn;t work reliably
def get_frequencies(vocab):
    # Identify frequent pairs in the vocabulary
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    # Merge the most frequent pair into a single token
    replacement = ''.join(pair)
    new_vocab = {}
    bigram = ' '.join(pair)

    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]

    return new_vocab

def tokenization(text, num_merges):
    # Clean up the input text first
    text = cleanup(text)
    vocab = {}

    # Tokenize the text, ensuring <newline>, <tab>, and other special tokens are treated as standalone
    for word in text.split():
        word = ' '.join(list(word)) + ' </w>'
        vocab[word] = vocab.get(word, 0) + 1

    # Apply Byte Pair Encoding (BPE) for the specified number of merges
    for i in range(num_merges):
        pairs = get_frequencies(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)

    return vocab


# Apply BPE
vocab = tokenization(all_text, NUM_MERGES)

# Save vocabulary to file
with open("vocab.txt", "w", encoding="utf-8") as file:
    for token in vocab.keys():
        file.write(f"{token}\n")
'''