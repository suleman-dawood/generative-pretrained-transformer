from constants import *
import torch
import torch.nn as nn

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        # these linear layers compute the key, query, and value for attention weight calculation
        # key, query, and value represent the components of the attention mechanism
        self.key = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)

        # this lower triangular matrix is used as a mask to prevent future tokens from being attended to in causal attention
        # the lower triangular matrix will ensure that each token only attends to itself and earlier tokens, not future ones
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

        # dropout layer for regularization, reducing overfitting by randomly setting parts of the attention weights to zero
        self.dropout = nn.Dropout(DROPOUT) 

    # forward pass through the attention calculations
    def forward(self, x):
        # compute the key, query, and value from the input tensor `x`
        key = self.key(x)  # applies linear transformation to input tokens to get the key
        query = self.query(x)  # applies linear transformation to input tokens to get the query
        value = self.value(x)  # applies linear transformation to input tokens to get the value

        # scale the key to stabilize the gradients during training (using the square root of the key dimension)
        scale_factor = key.shape[-1] ** -0.5 

        # calculate attention weights by computing the dot product of query and key, then scaling by the scale_factor
        # this determines how much attention each token should pay to other tokens
        attention_weights = query @ key.transpose(-2, -1) * scale_factor

        # apply the lower triangular mask to prevent future tokens from being attended to (causal attention)
        mask = self.tril[:context_size, :context_size]
        attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))

        # apply softmax to the attention weights to get probabilities, turning them into a distribution
        attention_weights = nn.functional.softmax(attention_weights, dim=1)

        # apply dropout to the attention weights for regularization
        attention_weights = self.dropout(attention_weights)

        # compute the output by multiplying the attention weights with the value vectors
        # this is how the model decides which tokens are important to attend to based on the learned attention weights
        output = attention_weights @ value

        # return the output of the attention mechanism
        return output
