import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================= Configuration =========================

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"Embedding dim ({config.n_embd}) must be divisible by n_heads ({config.n_head})"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # 768 / 12 = 64

        # Single linear layer that produces Q, K, V all at once
        # Input: (batch, seq_len, 768) → Output: (batch, seq_len, 2304) [= 3 × 768]
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Mixes information across heads after attention is computed
        # Input: (batch, seq_len, 768) → Output: (batch, seq_len, 768)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Not used directly in forward (Flash Attention handles it via is_causal=True),
        # but required for HuggingFace weight compatibility
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Single matrix multiply produces all three: (B, T, 768) → (B, T, 2304)
        qkv = self.c_attn(x)


        # Split the 2304-dim vector into three 768-dim vectors along the last dimension
        query, key, value = qkv.split(self.n_embd, dim=-1)
        # Each is now: (batch_size, seq_len, 768)


        # (B, T, 768) → (B, T, 12, 64) → transpose → (B, 12, T, 64)
        # This lets PyTorch compute attention for all 12 heads in parallel
        query = query.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key   = key.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Internally computes: softmax((Q @ K^T) / sqrt(head_dim)) @ V
        # is_causal=True applies the lower-triangular mask automatically
        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)


        # (B, 12, T, 64) → transpose → (B, T, 12, 64) → view → (B, T, 768)
        # contiguous() is needed because transpose changes memory layout,
        # and view() requires contiguous memory
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)

        return self.c_proj(attn_output)


# ========================= Feed-Forward Network =========================

class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        # Expands: (B, T, 768) → (B, T, 3072)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # GELU activation (Gaussian Error Linear Unit)
        # approximate="tanh" matches the original GPT-2 implementation
        self.gelu = nn.GELU(approximate="tanh")

        # Compresses back: (B, T, 3072) → (B, T, 768)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)      # Expand:   768 → 3072
        x = self.gelu(x)       # Activate: non-linear transformation
        x = self.c_proj(x)     # Compress: 3072 → 768
        return x


# ========================= Transformer Block =========================

class TransformerBlock(nn.Module):
    """One transformer block = Attention + FeedForward with residual connections.

    Architecture (Pre-Norm style, same as GPT-2):
        x ──┬── LayerNorm → Attention ──┐
            │                           + (residual add)
            └───────────────────────────┘
                ──┬── LayerNorm → FFN ──┐
                  │                     + (residual add)
                  └─────────────────────┘

    The residual connections (skip connections) allow gradients to flow
    directly backward through the network, preventing vanishing gradients
    in deep networks (12+ layers).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)   # Pre-attention normalization
        self.attn = CausalSelfAttention(config)     # Multi-head attention
        self.ln_2 = nn.LayerNorm(config.n_embd)    # Pre-FFN normalization
        self.ff = FeedForward(config)               # Feed-forward network

    def forward(self, x):
        # Attention sub-block with residual connection
        x = x + self.attn(self.ln_1(x))
        # Feed-forward sub-block with residual connection
        x = x + self.ff(self.ln_2(x))
        return x


# ========================= GPT Model =========================

class GPT(nn.Module):
    """The complete GPT-2 language model.

    Architecture overview:
        Input token IDs (B, T)
            ↓
        Token Embedding + Position Embedding  → (B, T, 768)
            ↓
        12 × TransformerBlock                 → (B, T, 768)
            ↓
        Final LayerNorm                       → (B, T, 768)
            ↓
        Linear Head (lm_head)                 → (B, T, 50257)  [logits over vocabulary]
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )

        # Language Model Head — projects hidden states to vocabulary logits
        # (B, T, 768) → (B, T, 50257)
        # Each output value is a score for a vocabulary token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: share the same weight matrix between token embeddings
        # and the output head. Intuition: if two tokens have similar embeddings,
        # they should also have similar output probabilities.
        # Saves ~38M parameters (768 × 50257).
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights using GPT-2's standard initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with normal distribution (std=0.02).

        This is called recursively on every sub-module via self.apply().
        - Linear layers: weights ~ N(0, 0.02), biases = 0
        - Embedding layers: weights ~ N(0, 0.02)
        - LayerNorm: uses PyTorch defaults (scale=1, shift=0) — not handled here
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        batch_size, seq_len = token_ids.size()
        assert seq_len <= self.config.block_size, (
            f"Cannot forward sequence of length {seq_len}, "
            f"block size is only {self.config.block_size}"
        )

        # Create position indices [0, 1, 2, ..., seq_len-1] on the same device
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=token_ids.device)

        # Look up embeddings: each token gets a "what" vector + a "where" vector
        token_embeddings = self.transformer.wte(token_ids)   # (B, T) → (B, T, 768)
        position_embeddings = self.transformer.wpe(position_ids)  # (T,) → (T, 768), broadcasts to (B, T, 768)
        x = token_embeddings + position_embeddings

        # Pass through all transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer normalization
        x = self.transformer.ln_f(x)

        # Project to vocabulary size to get per-token logits
        # (B, T, 768) → (B, T, 50257)
        logits = self.lm_head(x)

        # Compute loss if targets are provided (training mode)
        loss = None
        if targets is not None:
            # cross_entropy expects:
            #   input:  (N, C) where N = batch_size * seq_len, C = vocab_size
            #   target: (N,)   where each value is the correct token index
            # So we flatten: (B, T, 50257) → (B*T, 50257) and (B, T) → (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    # -------------------- Pretrained Weight Loading --------------------

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained GPT: {model_type}")

        # Architecture configs for each model size
        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),    # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),   # 350M params
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),   # 774M params
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M params
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)

        # Create our model with random weights
        model = GPT(config)
        our_state_dict = model.state_dict()

        # Filter out non-parameter buffer keys (the causal mask)
        our_keys = [k for k in our_state_dict.keys() if not k.endswith(".attn.bias")]

        # Download and load the HuggingFace model
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_state_dict = hf_model.state_dict()

        # Filter out HuggingFace buffer keys
        hf_keys = [k for k in hf_state_dict.keys() if not k.endswith(".attn.bias")]
        hf_keys = [k for k in hf_keys if not k.endswith(".attn.masked_bias")]

        # These weights need to be transposed because OpenAI used Conv1D (TF-style)
        # which stores weights as (in_features, out_features) — opposite of PyTorch's
        # nn.Linear which stores as (out_features, in_features)
        weights_to_transpose = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "ff.c_fc.weight",
            "ff.c_proj.weight",
        ]

        assert len(our_keys) == len(hf_keys), (
            f"Mismatched keys: {len(our_keys)} != {len(hf_keys)}"
        )

        # Copy weights from HuggingFace model to our model
        for key in hf_keys:
            if any(key.endswith(w) for w in weights_to_transpose):
                # Transpose Conv1D weights to match nn.Linear layout
                assert hf_state_dict[key].shape[::-1] == tuple(our_state_dict[key].shape)
                with torch.no_grad():
                    our_state_dict[key].copy_(hf_state_dict[key].t())
            else:
                # Direct copy for matching shapes
                assert hf_state_dict[key].shape == our_state_dict[key].shape
                with torch.no_grad():
                    our_state_dict[key].copy_(hf_state_dict[key])

        return model

    # -------------------- Optimizer Configuration --------------------

    def configure_optimizers(self, weight_decay, learning_rate, device):
        """Configure AdamW optimizer with proper weight decay separation.

        Weight decay (L2 regularization) should only be applied to weight
        matrices, NOT to biases or LayerNorm parameters. Applying decay to
        biases/norms can hurt training because these parameters serve
        different roles (shifting/scaling) and shouldn't be penalized for
        being large.

        We distinguish them by dimensionality:
            - dim >= 2 → weight matrices (apply decay)
            - dim <  2 → biases (1D) and LayerNorm scale/shift (1D) (no decay)
        """
        # Collect all trainable parameters with their names
        # self.named_parameters() comes from nn.Module — it recursively walks
        # every sub-module and yields (name, parameter_tensor) pairs
        trainable_params = {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad
        }

        # Separate into two groups based on tensor dimensionality
        params_with_decay = [p for n, p in trainable_params.items() if p.dim() >= 2]
        params_without_decay = [p for n, p in trainable_params.items() if p.dim() < 2]

        optimizer_param_groups = [
            {"params": params_with_decay, "weight_decay": weight_decay},
            {"params": params_without_decay, "weight_decay": 0.0},
        ]

        # Check if the current PyTorch version supports fused AdamW
        # Fused AdamW merges multiple GPU kernel launches into one = faster
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"

        optimizer = torch.optim.AdamW(
            optimizer_param_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),  # Momentum coefficients (standard GPT-2 values)
            eps=1e-8,            # Small constant to prevent division by zero
            fused=use_fused,
        )

        print(f"Using fused AdamW: {use_fused}")
        return optimizer
