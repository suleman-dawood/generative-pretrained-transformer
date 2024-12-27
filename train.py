import os
import torch
from constants import *
from model import *
from tokenizers import Tokenizer
from tqdm import tqdm

# load the tokenizer from the specified file
tokenizer = Tokenizer.from_file("vocab-vocab.json")

def load_and_encode_text(file_path, chunk_size=1024*1024):
    # reads and encodes text from a file in chunks
    file_size = os.path.getsize(file_path)  # get the file size in bytes
    num_chunks = file_size // chunk_size + 1  # calculate the number of chunks

    all_ids = []  # stores tokenized ids

    with open(file_path, 'r', encoding='utf-8-sig') as file:
        # iterate through file chunks using tqdm for progress display
        for _ in tqdm(range(num_chunks), desc="Processing file"):
            text = file.read(chunk_size)  # read a chunk of the file
            if not text:
                break  # stop if there's no more text to read
            encoded = tokenizer.encode(text)  # tokenize the text
            all_ids.extend(encoded.ids)  # append the token ids to the list

    return torch.tensor(all_ids, dtype=torch.long)  # return the ids as a torch tensor

def prepare_datasets(file_path):
    # prepares training and validation datasets by splitting the data
    data = load_and_encode_text(file_path)  # load and tokenize the file
    split_idx = int(DATA_SPLIT * len(data))  # calculate the split index
    print("prepared")
    return data[:split_idx], data[split_idx:]  # return training and validation splits

def get_batch(data, batch_size, context_size):
    # generates batches of input and target sequences from the data
    indices = torch.randint(len(data) - context_size, (batch_size,))  # random indices
    x = torch.stack([data[i:i + context_size] for i in indices])  # input sequences
    y = torch.stack([data[i + 1:i + context_size + 1] for i in indices])  # target sequences
    return x, y

@torch.no_grad()
def estimate_loss(model, training_set, validation_set):
    # evaluates the model's training and validation loss
    model.eval()  # set the model to evaluation mode
    losses = {'train': 0, 'val': 0}  # initialize loss dictionary
    for split, dataset in [('train', training_set), ('val', validation_set)]:
        batch_losses = []  # stores individual batch losses
        for _ in range(val_iterations):
            x, y = get_batch(dataset, batch_size, context_size)  # generate a batch
            logits, loss = model(x, y)  # forward pass
            batch_losses.append(loss.item())  # collect the batch loss
        losses[split] = sum(batch_losses) / len(batch_losses)  # average the batch losses
    model.train()  # set the model back to training mode
    return losses

def train_model(model, training_set, validation_set, epochs, checkpoint_dir, start_epoch=0):
    # trains the model and periodically saves checkpoints
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # initialize the optimizer
    for epoch in range(start_epoch, start_epoch + epochs):
        # estimate and print losses at validation intervals
        if epoch % val_iterations == 0:
            losses = estimate_loss(model, training_set, validation_set)
            print(f"Epoch {epoch}: Train Loss = {losses['train']:.4f}, Val Loss = {losses['val']:.4f}")

        # prepare a training batch
        x, y = get_batch(training_set, batch_size, context_size)
        logits, loss = model(x, y)  # forward pass and calculate loss
        optimizer.zero_grad()  # reset gradients
        loss.backward()  # backpropagate
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # gradient clipping
        optimizer.step()  # update model parameters

        # print progress for each epoch
        print(f"Epoch {epoch + 1}/{epochs + start_epoch} - Loss: {loss.item():.4f}")

        # save the model checkpoint periodically
        if epoch % SAVE_STEPS == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    return model  # return the trained model
