# gpt2-124M-pytorch

A clean, from-scratch implementation of OpenAI's **GPT-2 (124M)** language model in PyTorch — no frameworks, no abstractions, just raw transformer architecture.

Built for learning, experimenting, and understanding how large language models actually work under the hood.

---

## Architecture

GPT-2 is an **autoregressive transformer** — it predicts the next token in a sequence by attending to all previous tokens. The architecture follows a simple repeating pattern:

```
Input Text
    │
    ▼
┌──────────────────────────────┐
│   Token Embedding (50,257 → 768)   │  Maps each token ID to a 768-dim vector
│ + Position Embedding (1,024 → 768) │  Encodes the position of each token
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│     Transformer Block (×12)        │  Repeated 12 times
│  ┌────────────────────────┐        │
│  │ Layer Norm              │        │
│  │ Multi-Head Attention    │───┐    │
│  │ (12 heads, 64 dim each) │   │    │  Residual
│  └────────────────────────┘   +◄───│  Connection
│  ┌────────────────────────┐   │    │
│  │ Layer Norm              │   │    │
│  │ Feed-Forward Network    │───┘    │
│  │ (768 → 3072 → 768)     │        │
│  └────────────────────────┘        │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│   Final Layer Norm                 │
│   Linear Head (768 → 50,257)       │  Produces a score for every token
└──────────────────────────────┘     │  in the vocabulary
    │
    ▼
 Logits → Softmax → Next Token
```

### Key Components

| Component | What It Does |
|---|---|
| **Token Embedding** | Converts integer token IDs into 768-dimensional vectors that capture semantic meaning. |
| **Position Embedding** | Adds positional information since transformers have no inherent sense of token order. |
| **Multi-Head Causal Attention** | Each token attends to all previous tokens (not future ones) through 12 parallel attention heads. Uses Flash Attention for memory efficiency. |
| **Feed-Forward Network** | A two-layer MLP that independently transforms each token's representation. Expands to 4× the embedding size (3072), applies GELU activation, then compresses back. |
| **Residual Connections** | Skip connections around every attention and FFN block. These allow gradients to flow directly backward through the network, enabling training of deep (12+ layer) models. |
| **Weight Tying** | The token embedding matrix and the output projection share the same weights, saving ~38M parameters. |

### Model Specifications

| Parameter | Value |
|---|---|
| Parameters | 124M |
| Layers | 12 |
| Attention Heads | 12 |
| Embedding Dimension | 768 |
| Head Dimension | 64 |
| Feed-Forward Dimension | 3,072 |
| Context Window | 1,024 tokens |
| Vocabulary | 50,257 (GPT-2 BPE) |

---

## Training

The training pipeline includes several techniques used in production LLM training:

- **Cosine Learning Rate Schedule** with linear warmup — prevents destructive early updates and smoothly reduces learning rate over time
- **Mixed Precision Training** (bfloat16) — halves memory usage and doubles throughput on modern GPUs
- **Gradient Clipping** (max norm = 1.0) — prevents gradient explosions from destabilizing training
- **AdamW Optimizer** with weight decay separation — applies L2 regularization only to weight matrices, not biases or normalization parameters
- **Weight Tying** — shares parameters between the input embedding and output projection

---

## Project Structure

```
gpt2-124M-pytorch/
├── model.py          # Full GPT-2 architecture (Attention, FFN, TransformerBlock, GPT)
├── dataset.py        # Data loading and tokenization utilities
├── train.py          # Training loop with LR scheduling and mixed precision
├── generate.py       # Text generation with temperature and top-k sampling
├── requirements.txt  # Python dependencies
├── .gitignore        # Git ignore rules
└── input.txt         # Training corpus (not included — bring your own)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU (recommended, 4GB+ VRAM)
- Works on CPU too, just slower

### Installation

```bash
# Clone the repository
git clone https://github.com/PulKiTgUpTACoDe/LLM-Architecture-from-scratch.git

# Create a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate        # Linux/Mac
env\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Prepare Training Data

Place any `.txt` file as `input.txt` in the project root. Some options:

- [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1MB, good for testing)
- [OpenWebText](https://huggingface.co/datasets/openwebtext) (full GPT-2 training data)
- Any book, article collection, or text corpus

### Train from Scratch

```bash
python train.py
```

You'll see output like:

```
Using device: cuda
Loaded 338,025 tokens from 'input.txt'
1 epoch = 330 batches
Using fused AdamW: True
step    0 | loss 10.9543 | lr 3.0000e-05 | dt 1423.17ms
step    1 | loss 10.8921 | lr 6.0000e-05 | dt 312.45ms
...
step   49 | loss  6.1234 | lr 3.0000e-05 | dt 305.12ms
```

Edit the hyperparameters at the top of `train.py` to adjust batch size, sequence length, learning rate, and number of steps.

### Generate Text

```bash
# Using pretrained GPT-2 weights (downloads from HuggingFace)
python generate.py --pretrained --prompt "The meaning of life is"

# Using your own trained checkpoint
python generate.py --checkpoint model.pt --prompt "Once upon a time"

# Adjust creativity
python generate.py --pretrained --prompt "In the year 2050" --temperature 1.2 --top_k 40 --max_tokens 200
```

**Sampling parameters:**

| Flag | Default | Effect |
|---|---|---|
| `--temperature` | 0.8 | Lower = more focused/deterministic, Higher = more creative/random |
| `--top_k` | 50 | Only sample from the top-k most likely next tokens |
| `--max_tokens` | 100 | Number of tokens to generate |

---

## Acknowledgements

This implementation is based on Andrej Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt) lecture series and Sebastian Raschka's [Build a Large Language Model From Scratch](https://github.com/rasbt/LLMs-from-scratch).
