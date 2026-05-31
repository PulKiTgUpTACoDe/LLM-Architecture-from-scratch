import math
import time

import torch

from model import GPT, GPTConfig
from dataset import DataLoader


# ========================= Hyperparameters =========================

BATCH_SIZE = 4       # Number of sequences processed in parallel per step
SEQ_LEN = 256        # Length of each sequence (context window for training)

MAX_LR = 3e-4        # Peak learning rate (standard for GPT-2)
MIN_LR = MAX_LR * 0.1  # Floor learning rate (10% of peak)
WARMUP_STEPS = 10    # Steps to linearly ramp up LR from near-zero to MAX_LR
MAX_STEPS = 50       # Total training steps (increase for real training)

WEIGHT_DECAY = 0.1   # L2 regularization strength (only applied to weight matrices)


# ========================= Learning Rate Schedule =========================

def get_lr(step):
    """Cosine decay learning rate schedule with linear warmup.

    Schedule visualization:
        LR          
         ↑     ╱‾‾‾╲
         |    ╱      ╲
         |   ╱        ╲___________
         |  ╱
         +───────────────────────→ steps
           ↑          ↑           ↑
          0      warmup=10    max_steps=50

    Phase 1 (steps 0–9):   Linear warmup   — LR increases from ~0 to MAX_LR
    Phase 2 (steps 10–50): Cosine decay    — LR smoothly decreases to MIN_LR
    Phase 3 (steps 50+):   Constant floor  — LR stays at MIN_LR
    """
    # Phase 1: Linear warmup
    # At step 0: lr = 3e-4 * (1/10) = 3e-5  (start small)
    # At step 9: lr = 3e-4 * (10/10) = 3e-4 (reach peak)
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS

    # Phase 3: After all decay steps, hold at minimum
    if step > MAX_STEPS:
        return MIN_LR

    # Phase 2: Cosine decay
    # decay_ratio goes from 0.0 (step=10) to 1.0 (step=50)
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1

    # cos(0) = 1 → coeff = 1.0 → LR = MAX_LR  (start of decay)
    # cos(π) = -1 → coeff = 0.0 → LR = MIN_LR (end of decay)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return MIN_LR + (MAX_LR - MIN_LR) * coeff


# ========================= Training Loop =========================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Data ---
    train_loader = DataLoader(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

    # --- Model ---
    model = GPT(GPTConfig())
    model.to(device)

    # --- Optimizer ---
    # Uses AdamW with separate weight decay groups (matrices vs biases/norms)
    optimizer = model.configure_optimizers(
        weight_decay=WEIGHT_DECAY, learning_rate=MAX_LR, device=device
    )

    # --- Training ---
    for step in range(MAX_STEPS):
        step_start_time = time.time()

        # 1. Fetch batch of inputs and targets
        inputs, targets = train_loader.next_batch()
        inputs, targets = inputs.to(device), targets.to(device)

        # 2. Clear gradients from previous step
        #    Without this, PyTorch would accumulate (add) new gradients on top of old ones
        optimizer.zero_grad()

        # 3. Forward pass with mixed precision (bfloat16)
        #    autocast tells PyTorch to run matrix multiplications in 16-bit
        #    instead of 32-bit wherever safe. This halves memory usage and
        #    roughly doubles speed on modern GPUs.
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(inputs, targets)

        # 4. Backward pass (backpropagation)
        #    Computes the gradient of the loss with respect to every parameter
        loss.backward()

        # 5. Gradient clipping
        #    Computes the L2 norm of ALL gradients across the model.
        #    If the total norm exceeds 1.0, scales all gradients down proportionally.
        #    Prevents a single bad batch from causing a catastrophic weight update.
        #    Example: if grad norm = 5.0, all grads are multiplied by (1.0 / 5.0) = 0.2
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 6. Update learning rate according to cosine schedule
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # 7. Optimizer step — apply gradients to update model weights
        optimizer.step()

        # --- Logging ---
        step_time_ms = (time.time() - step_start_time) * 1000
        print(
            f"step {step:4d} | "
            f"loss {loss.item():.4f} | "
            f"lr {current_lr:.4e} | "
            f"dt {step_time_ms:.2f}ms"
        )


if __name__ == "__main__":
    train()
