from head import Head
from constants import *
import torch.nn as nn
import torch

# defines a module for multiple attention heads running in parallel
class MultiHead(nn.Module):

    def __init__(self, head_count, head_size):
        super().__init__()
        # create a list of attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_count)])
        # linear layer to project concatenated outputs into an embedding vector
        self.proj = nn.Linear(head_size * head_count, num_embeddings)
        self.dropout = nn.Dropout(DROPOUT)  # apply dropout for regularization

    def forward(self, x):
        head_outputs = []
        # apply each attention head to the input
        for head in self.heads:
            head_outputs.append(head(x))
        # concatenate outputs along the last dimension
        concatenated = torch.cat(head_outputs, dim=-1)
        # project concatenated outputs and apply dropout
        return self.dropout(self.proj(concatenated))
