from tokenizers import Tokenizer, processors, pre_tokenizers, decoders, trainers, models
from constants import *
from preprocess import *
import os

def train_bpe_tokenizer():
    # initialize the tokenizer with the bpe model
    tokenizer = Tokenizer(models.BPE())

    # set up pre-tokenization and decoding methods
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # configure the bpe trainer with specific settings
    trainer = trainers.BpeTrainer(
        vocab_size=NUM_MERGES,  # maximum vocabulary size
        min_frequency=2,  # minimum frequency for merges
        show_progress=True,  # display progress during training
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # initialize with byte-level alphabet
    )

    # define paths for the input and output text files
    file = os.path.join("pretrain_data", "train.txt")  # original text file
    file2 = os.path.join("pretrain_data", "processed.txt")  # preprocessed file

    num_lines = 0
    # preprocess the input file and write to the output file
    with open(file2, 'w', encoding='utf-8') as processed_file:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                num_lines += 1
                # log progress every NUM_LINE_SPLIT lines
                if num_lines % NUM_LINE_SPLIT == 0:
                    print(f"Processing line {num_lines}")

                # preprocess the current line (keep special characters)
                preprocessed_line = preprocess_text(line)

                # write the preprocessed line to the output file
                processed_file.write(preprocessed_line + '\n')  # ensure newline after each line

    print("Preprocessing complete. all preprocessed text saved to processed.txt.")

    # train the tokenizer on the preprocessed file
    tokenizer.train([file2], trainer=trainer)

    return tokenizer  # return the trained tokenizer

def save_tokenizer(tokenizer, vocab_file='vocab-vocab.json'):
    # save the trained tokenizer to a file
    tokenizer.save(vocab_file)
    print(f"Tokenizer saved to {vocab_file}")

def load_tokenizer(vocab_file='vocab-vocab.json'):
    # load a tokenizer from a previously saved file
    tokenizer = Tokenizer.from_file(vocab_file)
    print(f"Tokenizer loaded from {vocab_file}")
    return tokenizer

# train and save the tokenizer
tokenizer = train_bpe_tokenizer()
save_tokenizer(tokenizer)
