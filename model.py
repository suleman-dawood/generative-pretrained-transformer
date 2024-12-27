import torch.nn as nn
import torch
import block
from constants import *

class BigramModel(nn.Module):

    def __init__(self):
        super().__init__()
        # embedding for tokens, maps tokens into dense vectors capturing semantic info
        self.token_embedding_table = nn.Embedding(NUM_MERGES, num_embeddings)
        # embedding for positions, maps positions into dense vectors to capture positional info
        self.position_embedding_table = nn.Embedding(context_size, num_embeddings)

        # stack of transformer blocks
        blocks = [block.Block(num_embeddings, head_count) for _ in range(layer_count)]
        self.blocks = nn.Sequential(*blocks)

        self.ln1 = nn.LayerNorm(num_embeddings)  # normalizes activations across embeddings
        self.linear_projection = nn.Linear(num_embeddings, NUM_MERGES)  # maps embeddings into logits
        self.apply(self._init_weights)  # initialize weights for all layers
        self.loss_fn = nn.CrossEntropyLoss()  # loss function for training

    def _init_weights(self, module):
        # initializes weights based on layer type
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, index, targets=None):
        # token embeddings represent input tokens as dense vectors
        token_embeddings = self.token_embedding_table(index)
        # position embeddings add positional info
        batch_length, seq_length = index.shape
        position_indices = torch.arange(seq_length)  # sequence length
        position_embeddings = self.position_embedding_table(position_indices).unsqueeze(0)  # batch dimension
        # combine token and position embeddings
        combined_embeddings = token_embeddings + position_embeddings
        # pass through transformer blocks and project to logits
        logits = self.linear_projection(self.blocks(self.ln1(combined_embeddings)))

        if targets is None:
            loss = None
        else:
            # resize logits and targets for compatibility
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = self.loss_fn(logits, targets)

        return logits, loss

    def generate(self, index, new_tokens):
        # generates new tokens based on the current sequence
        for i in range(new_tokens):
            logits, loss = self(index[:, -context_size:])  # use only the last context_size tokens
            logits = logits[:, -1, :]  # predictions for the last token
            probs = nn.functional.softmax(logits, dim=-1)  # probability distribution over vocab
            next_index = torch.multinomial(probs, num_samples=1)  # sample the next token
            index = torch.cat((index, next_index), dim=1)  # append sampled token to sequence
        return index
