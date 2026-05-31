"""
Data loading and tokenization utilities for GPT-2 training.

Contains:
    DataLoader      → Reads a text file, tokenizes it, and serves sequential batches
    text_to_tokens  → Encode a string → tensor of token IDs
    tokens_to_text  → Decode a tensor of token IDs → string
"""

import torch
import tiktoken


# ========================= Data Loader =========================

class DataLoader:
    """Simple sequential data loader for GPT-2 training.

    How it works:
        1. Reads the entire text file into memory.
        2. Tokenizes it into a single flat 1D tensor of token IDs.
        3. On each call to next_batch(), slices out a chunk and reshapes it
           into (batch_size, seq_len) input-target pairs.

    The target is the input shifted by one token (next-token prediction):
        text:    "The cat sat on"
        input:   [The, cat, sat]
        target:  [cat, sat, on]
    """

    def __init__(self, batch_size, seq_len, file_path="input.txt"):
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Tokenize the entire text corpus
        encoder = tiktoken.get_encoding("gpt2")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        token_ids = encoder.encode(raw_text)
        self.tokens = torch.tensor(token_ids, dtype=torch.long)

        print(f"Loaded {len(self.tokens):,} tokens from '{file_path}'")
        print(f"1 epoch = {len(self.tokens) // (batch_size * seq_len):,} batches")

        # Pointer tracking which token we're currently at
        self.current_position = 0

    def next_batch(self):
        """Return the next (input, target) batch and advance the pointer.

        Returns:
            inputs:  (batch_size, seq_len) tensor of input token IDs
            targets: (batch_size, seq_len) tensor of target token IDs

        Example with batch_size=2, seq_len=3:
            Flat tokens: [10, 20, 30, 40, 50, 60, 70]
            Buffer (B*T+1 = 7 tokens): [10, 20, 30, 40, 50, 60, 70]

            inputs  = [10, 20, 30, 40, 50, 60].view(2,3) = [[10, 20, 30],
                                                              [40, 50, 60]]

            targets = [20, 30, 40, 50, 60, 70].view(2,3) = [[20, 30, 40],
                                                              [50, 60, 70]]

            Notice: targets[i][j] = inputs[i][j+1] (the next token)
        """
        batch_size, seq_len = self.batch_size, self.seq_len
        tokens_needed = batch_size * seq_len + 1  # +1 for the final target token

        # If not enough tokens remain, wrap around to the beginning of the corpus
        if self.current_position + tokens_needed > self.tokens.size(0):
            self.current_position = 0

        # Slice a flat buffer of (B * T + 1) tokens
        buffer = self.tokens[self.current_position : self.current_position + tokens_needed]

        # Split into inputs and targets (shifted by 1 token)
        inputs  = buffer[:-1].view(batch_size, seq_len)
        targets = buffer[1:].view(batch_size, seq_len)

        # Advance the pointer
        self.current_position += batch_size * seq_len
        return inputs, targets


# ========================= Tokenization Helpers =========================

# Module-level encoder (initialized once, reused across calls)
_encoder = tiktoken.get_encoding("gpt2")


def text_to_tokens(text):
    """Encode a string into a batched tensor of token IDs.

    Args:
        text: A plain text string, e.g. "Hello, I am"

    Returns:
        Tensor of shape (1, num_tokens), e.g. tensor([[15496, 11, 314, 716]])
        The leading dimension of 1 is the batch dimension (single sequence).
    """
    token_ids = _encoder.encode(text)
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # Add batch dim


def tokens_to_text(token_ids):
    """Decode a tensor of token IDs back into a readable string.

    Args:
        token_ids: Tensor of shape (1, seq_len) or (seq_len,)

    Returns:
        Decoded string, e.g. "Hello, I am"
    """
    if token_ids.dim() == 2:
        token_ids = token_ids.squeeze(0)  # Remove batch dimension
    return _encoder.decode(token_ids.tolist())
