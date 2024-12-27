import os
from datasets import load_dataset
from tqdm import tqdm
from preprocess import *

# constants
DATASET_NAME = "AiresPucrs/stanford-encyclopedia-philosophy"  # name of the dataset from hugging face
INITIAL_DIRECTORY = "pretrain_data"  # directory where the processed dataset will be saved
OUTPUT_FILE = "train.txt"  # output filename for processed data

# ensure the output directory exists
os.makedirs(INITIAL_DIRECTORY, exist_ok=True)  # creates the directory if it does not exist

def process_dataset(dataset, output_file):
    file_path = os.path.join(INITIAL_DIRECTORY, output_file)  # define the full path for the output file

    # open the file for writing
    with open(file_path, "w", encoding="utf-8") as text_file:
        # iterate over the dataset and write each entry to the file
        for example in tqdm(dataset, desc="processing entries", unit="entry"):
            text = example["text"]  # extract the text from each entry
            text = preprocess_text(text)
            text_file.write(text + "\n")  # write the text followed by a newline

    print(f"processing complete. processed data saved to {file_path}")  # notify when processing is complete
    return file_path  # return the path of the saved file

# user prompt to decide whether to download and process the dataset
user_input = input("do you wish to download and process the dataset? (y/n): ").strip().lower()
if user_input == "y":
    # load the dataset from hugging face's repository
    dataset = load_dataset(DATASET_NAME, split="train")
    # process the dataset and save it to the specified file
    processed_file_path = process_dataset(dataset, OUTPUT_FILE)
    print(f"dataset processed and saved at {processed_file_path}")  # print the location of the processed file
else:
    print("dataset processing aborted.")  # print if the user decides not to proceed
