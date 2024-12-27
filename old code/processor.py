import os
import torch
from tqdm import tqdm
from tokenizers import Tokenizer

# my pc couldn't handle the huge data
def process_file(file_path, encode, chunk_size=512*512, output_file='data_tensor.pt'):
    # Ensure the output file does not exist
    if os.path.exists(output_file):
        os.remove(output_file)

    # Get the total file size for progress tracking
    total_size = os.path.getsize(file_path)

    with open(file_path, 'r', encoding='utf-8') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc='Processing') as pbar:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            # Encode the chunk and convert to tensor
            encoded_chunk = encode(chunk)
            tensor_chunk = torch.tensor(encoded_chunk, dtype=torch.long)

            # Append the tensor to the output file
            with open(output_file, 'ab') as out_f:
                torch.save(tensor_chunk, out_f)

            # Update the progress bar
            pbar.update(len(chunk.encode('utf-8')))

    print(f"Final tensor saved to {output_file}")


tokenizer = Tokenizer.from_file("vocab-vocab.json")

# Function to encode the tokens
def encode(text):
    encoded = tokenizer.encode(text)
    return encoded.ids  # Return the token indices


# Example usage
file_path = os.path.join("pretrain_data", "processed.txt")
process_file(file_path, encode)

