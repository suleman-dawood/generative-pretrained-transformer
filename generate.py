from model import BigramModel
import torch
from constants import *
from tokenizers import Tokenizer
import re

def post_process(text):
    # Remove spaces before punctuation (like commas, periods, etc.)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Remove spaces before and after brackets (parentheses, curly brackets, etc.)
    text = re.sub(r'\s*([<>(){}\[\]=])\s*', r'\1', text)
    # Ensure single space after punctuation if followed by a word character
    text = re.sub(r'([.,!?;:])(\w)', r'\1 \2', text)
    return text

def generate_text(out_length):
    # Initialize the model and load the pre-trained weights
    sample_model = BigramModel()
    sample_model.load_state_dict(torch.load("trained_model.pth"))  # Load trained model weights
    sample_model.eval()  # Set the model to evaluation mode

    # Initialize the tokenizer from the vocab file
    tokenizer = Tokenizer.from_file("vocab-vocab.json")

    # Generate sample text with an initial context of zeros (empty context)
    initial = torch.zeros((1, context_size), dtype=torch.long)
    # Generate new tokens (length specified by `out_length`)
    sample_generation = sample_model.generate(initial, new_tokens=out_length)[0].tolist()

    # Decode the generated token indices into human-readable text
    decoded_text = tokenizer.decode(sample_generation)
    lines = decoded_text.split('\n')

    # If the generated text contains multiple lines, remove the first line (which may contain metadata or headers)
    if len(lines) > 1:
        decoded_text = '\n'.join(lines[1:]).strip()

    # Post-process the text (remove extra spaces, format punctuation, etc.)
    decoded_text = post_process(decoded_text)

    # Write the final processed text to an output file
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(decoded_text)

    return decoded_text  # Return the generated text

# Generate text of specified length (300 tokens)
generate_text(300)
