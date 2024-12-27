import os
import re
from constants import *

def preprocess_text(text):
    # remove urls and email addresses
    text = re.sub(r'http\S+|www\S+', '', text)  # remove urls
    text = re.sub(r'\S+@\S+', '', text)  # remove email addresses
    text = re.sub(r'<.*?>', '', text)  # remove html tags if present

    # normalize quotes
    text = re.sub(r"[‘’“”]", "'", text)  # convert quotes to standard single quotes
    text = re.sub(r'([.,!?;:])', r" \1 ", text)  # add spaces around punctuation
    text = re.sub(r"\b(\w+)'(\w+)\b", r"\1'\2", text)  # preserve contractions
    text = re.sub(r'[-_—]', '', text)  # remove dashes
    text = re.sub(r'([<>(){}=[\]])', r' \1 ', text)  # add spaces around brackets
    text = text.replace("\ufeff", "")  # remove bom if present

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # split the text into words
    words = text.split()

    # filter out non-ascii words
    words = [word for word in words if all(ord(char) < 128 for char in word)]

    # rejoin the words into a single string
    return " ".join(words)

'''
if os.path.exists(file):
    num_lines = 0
    num_words = 0
    num_characters = 0
    unique_tokens = set()  # use a set to store unique tokens

    # open the file and process it line by line
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            num_lines += 1
            # print progress every NUM_LINE_SPLIT lines
            if num_lines % NUM_LINE_SPLIT == 0:
                print(f"processing line {num_lines}")

            # stop processing after reaching a limit
            if num_lines > NUM_LINES:
                print("reached 10 million lines. stopping processing.")
                break

            # preprocess the line (retain special characters)
            preprocessed_line = preprocess_text(line)

            # tokenize the preprocessed line
            tokens = preprocessed_line.split()  # byte-level tokenization will be done later

            num_words += len(tokens)
            num_characters += len(preprocessed_line)

            # update the set of unique tokens
            unique_tokens.update(tokens)

    # compute vocabulary size
    vocab_size = len(unique_tokens)

    print(f"file stats (preprocessed):")
    print(f"number of lines: {num_lines}")
    print(f"number of words: {num_words}")
    print(f"number of characters: {num_characters}")
    print(f"vocabulary size (unique tokens): {vocab_size}")
else:
    print("the file does not exist.")
'''
