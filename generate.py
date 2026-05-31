"""
Text generation script for GPT-2.

Supports three modes:
    python generate.py                                  # Random weights (gibberish)
    python generate.py --pretrained                     # HuggingFace pretrained GPT-2
    python generate.py --checkpoint model.pt            # Your own trained checkpoint

Sampling options:
    --prompt "Once upon a time"     Starting text
    --max_tokens 200                How many tokens to generate
    --temperature 0.8               Randomness (0.1 = focused, 2.0 = wild)
    --top_k 50                      Only sample from top 50 most likely tokens
"""

import argparse

import torch

from model import GPT, GPTConfig
from dataset import text_to_tokens, tokens_to_text


# ========================= Generation Logic =========================

def generate(model, token_ids, max_new_tokens, context_size, temperature=1.0, top_k=None):
    """Auto-regressively generate tokens from a trained GPT model.

    How it works (repeated max_new_tokens times):
        1. Crop the input to the model's max context size
        2. Forward pass to get logits for the next token
        3. Apply temperature scaling to control randomness
        4. Apply top-k filtering to remove unlikely tokens
        5. Sample from the resulting probability distribution
        6. Append the sampled token and repeat

    Args:
        model:          A GPT model instance
        token_ids:      Starting token indices, shape (batch_size, seq_len)
        max_new_tokens: Number of new tokens to generate
        context_size:   Maximum context window the model supports (1024 for GPT-2)
        temperature:    Controls output randomness:
                            < 1.0 → more deterministic, picks high-probability tokens
                            = 1.0 → unchanged (default)
                            > 1.0 → more random, flattens the probability distribution
        top_k:          If set, only consider the top-k most likely next tokens.
                        All other tokens are masked out before sampling.

    Returns:
        token_ids tensor including the original prompt + generated continuation
    """
    model.eval()

    for _ in range(max_new_tokens):
        # Crop context if it exceeds the model's maximum sequence length
        # If we've generated 1100 tokens but the model only supports 1024,
        # we keep only the last 1024 tokens as input
        context = token_ids[:, -context_size:]

        # Forward pass — get predictions without computing gradients
        with torch.no_grad():
            logits, _ = model(context)

        # Extract logits for the LAST token position only
        # logits shape: (batch, seq_len, vocab_size) → (batch, vocab_size)
        next_token_logits = logits[:, -1, :]

        # Temperature scaling: divide logits by temperature before softmax
        # Low temp (0.2):  logits become [10, 1] → [50, 5] → softmax ≈ [1.0, 0.0] (confident)
        # High temp (2.0): logits become [10, 1] → [5, 0.5] → softmax ≈ [0.6, 0.4] (uncertain)
        next_token_logits = next_token_logits / temperature

        # Top-K filtering: zero out all tokens outside the top-k most likely
        if top_k is not None:
            # Find the k-th largest logit value as a threshold
            top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            threshold = top_k_values[:, [-1]]  # Smallest value in the top-k

            # Set all logits below the threshold to -infinity
            # After softmax, e^(-inf) = 0, so these tokens get 0% probability
            next_token_logits[next_token_logits < threshold] = -float("Inf")

        # Convert logits to probabilities and sample
        probabilities = torch.softmax(next_token_logits, dim=-1)

        # Multinomial sampling: randomly pick a token weighted by probability
        # Unlike argmax (always picks the top token), this allows diversity
        sampled_token = torch.multinomial(probabilities, num_samples=1)

        # Append the new token to the running sequence
        token_ids = torch.cat((token_ids, sampled_token), dim=1)

    return token_ids


# ========================= CLI Entry Point =========================

def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT-2")
    parser.add_argument("--prompt", type=str, default="Hello, I am",
                        help="Starting text prompt")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.1=focused, 2.0=wild)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Only sample from top-k most likely tokens")
    parser.add_argument("--pretrained", action="store_true",
                        help="Load pretrained GPT-2 weights from HuggingFace")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a local .pt checkpoint file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    if args.pretrained:
        model = GPT.from_pretrained("gpt2")
    elif args.checkpoint:
        model = GPT(GPTConfig())
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("Warning: Using randomly initialized weights. Output will be gibberish.")
        model = GPT(GPTConfig())

    model.to(device)
    model.eval()

    # --- Encode Prompt ---
    prompt_tokens = text_to_tokens(args.prompt).to(device)

    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)

    # --- Generate ---
    output_tokens = generate(
        model,
        prompt_tokens,
        max_new_tokens=args.max_tokens,
        context_size=GPTConfig.block_size,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_text = tokens_to_text(output_tokens)
    print(generated_text)


if __name__ == "__main__":
    main()
