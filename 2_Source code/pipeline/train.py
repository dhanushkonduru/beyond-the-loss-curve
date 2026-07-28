"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 3 — Training (fine-tune on full TOFU: retain + forget)
=============================================================

Trains the from-scratch GPT on all 4000 TOFU QA pairs so it
"knows" all 200 fictional authors. Phase 4 will then unlearn
the 20 forget-set authors.

  - AdamW, lr 5e-4, linear warmup + cosine decay
  - mixed precision (fp16) on GPU for T4 speed
  - gradient clipping at 1.0
  - checkpoints -> Google Drive on Colab, ./checkpoints locally
  - post-training spot check: asks about forget & retain authors

Run:  python train.py
"""

import math
import os
import time

import torch

from model import GPT, PRESETS, DEVICE
from data import get_tokenizer, make_dataloaders

# -------------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------------
PRESET = "small-50m"
# from-scratch model: if the spot check looks weak, raise this (e.g. EPOCHS=20 python train.py)
EPOCHS = int(os.environ.get("EPOCHS", 5))
SEED = int(os.environ.get("SEED", 42))   # fixed seed -> reproducible runs
LR = 5e-4
WEIGHT_DECAY = 0.1
WARMUP_FRAC = 0.05    # first 5% of steps are linear warmup
GRAD_CLIP = 1.0


# -------------------------------------------------------------
# Checkpoint directory: Google Drive on Colab, local otherwise
# -------------------------------------------------------------
def get_checkpoint_dir():
    try:
        from google.colab import drive  # only exists on Colab
        drive.mount("/content/drive")
        ckpt_dir = "/content/drive/MyDrive/forget-me-not/checkpoints"
        print("[Phase 3] Colab detected — checkpoints go to Google Drive")
    except ImportError:
        ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        print("[Phase 3] Not on Colab — checkpoints go to ./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def save_checkpoint(model, path, extra=None):
    payload = {"model_state": model.state_dict(), "preset": PRESET}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[Phase 3] Checkpoint saved -> {path}")


def load_checkpoint(path, device=DEVICE):
    """Used by Phases 4 and 5 to reload trained models."""
    payload = torch.load(path, map_location=device)
    model = GPT(PRESETS[payload["preset"]]).to(device)
    model.load_state_dict(payload["model_state"])
    return model


# -------------------------------------------------------------
# LR schedule: linear warmup then cosine decay
# -------------------------------------------------------------
def lr_at(step, total_steps):
    warmup = max(1, int(total_steps * WARMUP_FRAC))
    if step < warmup:
        return LR * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 * LR + 0.9 * LR * 0.5 * (1 + math.cos(math.pi * progress))


# -------------------------------------------------------------
# Training loop
# -------------------------------------------------------------
def train(model, loader, epochs=EPOCHS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                                  betas=(0.9, 0.95))
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    total_steps = epochs * len(loader)
    print(f"[Phase 3] Training: {epochs} epochs x {len(loader)} batches "
          f"= {total_steps} steps (amp={'on' if use_amp else 'off'})")

    model.train()
    step, t0 = 0, time.time()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # set scheduled learning rate
            lr = lr_at(step, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                _, loss = model(input_ids, targets=labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"[Phase 3]   step {step:>5}/{total_steps} | "
                      f"loss {loss.item():.3f} | lr {lr:.2e} | "
                      f"{time.time() - t0:.0f}s elapsed")

        print(f"[Phase 3] Epoch {epoch}/{epochs} done — avg loss {epoch_loss / len(loader):.3f}")
    return model


# -------------------------------------------------------------
# Ask the model a question (also used by Phase 5's audit)
# -------------------------------------------------------------
@torch.no_grad()
def ask_model(model, tokenizer, question, max_new_tokens=60):
    prompt = f"Question: {question} Answer:"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    out = model.generate(ids, max_new_tokens=max_new_tokens,
                         temperature=1.0, top_k=1,          # top_k=1 -> greedy, deterministic
                         eos_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(out[0][ids.shape[1]:])
    return answer.replace(tokenizer.eos_token, "").strip()


def spot_check(model, tokenizer, datasets, n=3):
    """Ask about a few forget-set and retain-set authors to confirm learning."""
    print("\n[Phase 3] ---- SPOT CHECK: did the model learn TOFU? ----")
    for split in ("forget", "retain"):
        print(f"\n  Split: {split.upper()}")
        for i in range(n):
            ex = datasets[split].examples[i * 20]   # first question of author i
            reply = ask_model(model, tokenizer, ex["question"])
            print(f"  Q: {ex['question']}")
            print(f"  model : {reply[:150]}")
            print(f"  truth : {ex['answer'][:150]}\n")
    print("[Phase 3] If replies resemble the truth, training worked. "
          "If they're gibberish, raise EPOCHS and retrain.")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(SEED)
    ckpt_dir = get_checkpoint_dir()
    tokenizer = get_tokenizer()
    loaders, datasets = make_dataloaders(tokenizer)

    print(f"[Phase 3] Building fresh model (preset: {PRESET}, seed: {SEED})...")
    model = GPT(PRESETS[PRESET]).to(DEVICE)

    model = train(model, loaders["full"])

    ckpt_path = os.path.join(ckpt_dir, "trained_model.pt")
    save_checkpoint(model, ckpt_path, extra={"phase": "trained", "epochs": EPOCHS})

    spot_check(model, tokenizer, datasets)
    print("\n[Phase 3] ✅ Training complete. Next: Phase 4 (Gradient Ascent unlearning).")
