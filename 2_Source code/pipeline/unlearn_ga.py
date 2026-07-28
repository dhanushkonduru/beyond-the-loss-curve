"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 4 — Unlearning via Gradient Ascent
=============================================================

Idea: training MINIMIZES loss on data; to forget, we MAXIMIZE
loss on the forget set (equivalently, minimize the NEGATIVE loss).
This is the simplest unlearning baseline in the literature.

Danger: gradient ascent is a blunt instrument — run it too long
or too hot and the model forgets EVERYTHING (catastrophic collapse).
Safety measures here:
  - much smaller LR than training (1e-5 vs 5e-4)
  - hard cap on steps (default 150, per your 100-200 spec)
  - live monitoring every 5 steps: forget-loss (should RISE) vs
    retain-loss (should stay flat)
  - early stop when forget-loss passes a target
  - ROLLBACK: if retain-loss drifts too far (collapse), restore the
    last snapshot where the model was still healthy

Requires: checkpoints/trained_model.pt from Phase 3.
Run:  python unlearn_ga.py
"""

import os

import torch

from model import DEVICE
from data import get_tokenizer, make_dataloaders
from train import ask_model, get_checkpoint_dir, load_checkpoint, save_checkpoint

# -------------------------------------------------------------
# Unlearning hyperparameters
# -------------------------------------------------------------
GA_LR = 3e-5              # smaller than training LR — GA on an overfit model
                          # explodes fast (Adam momentum feeds back), so go gentle
MAX_STEPS = 150           # hard cap (your spec: 100-200)
GRAD_CLIP = 1.0
LOG_EVERY = 5             # health-check often: collapse can happen within 10 steps

# stopping conditions
FORGET_LOSS_TARGET = 3.0  # vs ~0.000 after training — answers thoroughly scrambled
RETAIN_DRIFT_LIMIT = 2.0  # GA has no anchor, so on templated WHO data it damages retain
                          # freely; loose limit lets it reach the target (the audit shows the cost)
EVAL_BATCHES = 20         # batches used for the retain-loss health check


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
@torch.no_grad()
def eval_loss(model, loader, max_batches=EVAL_BATCHES):
    """Average loss over a few batches — quick health thermometer."""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        _, loss = model(batch["input_ids"].to(DEVICE), targets=batch["labels"].to(DEVICE))
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def forever(loader):
    """Cycle a DataLoader endlessly (forget set is only 25 batches)."""
    while True:
        for batch in loader:
            yield batch


# -------------------------------------------------------------
# Gradient Ascent unlearning
# -------------------------------------------------------------
def unlearn(model, forget_loader, retain_loader, max_steps=MAX_STEPS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=GA_LR)

    # baselines before we touch anything
    base_forget = eval_loss(model, forget_loader)
    base_retain = eval_loss(model, retain_loader)
    print(f"[Phase 4] Baseline losses — forget: {base_forget:.3f}, retain: {base_retain:.3f}")
    print(f"[Phase 4] Goal: push forget-loss UP toward {FORGET_LOSS_TARGET}, "
          f"keep retain-loss near {base_retain:.3f}")

    model.train()
    batches = forever(forget_loader)
    stop_reason = f"reached step cap ({max_steps})"
    # rollback snapshot: the last weights where retain-loss was still healthy.
    # GA can explode between two health checks; without this, stopping "on"
    # the drift limit still saves an already-damaged model.
    healthy_state = None

    for step in range(1, max_steps + 1):
        batch = next(batches)
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets=labels)

        # THE core trick: negate the loss -> gradient ASCENT on forget data
        (-loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            forget_now = eval_loss(model, forget_loader)
            retain_now = eval_loss(model, retain_loader)
            print(f"[Phase 4]   step {step:>3}/{max_steps} | "
                  f"forget-loss {forget_now:.3f} (▲ from {base_forget:.3f}) | "
                  f"retain-loss {retain_now:.3f} (baseline {base_retain:.3f})")

            if retain_now - base_retain > RETAIN_DRIFT_LIMIT:
                stop_reason = (f"retain-loss drifted {retain_now - base_retain:.2f} above baseline "
                               f"— collapse detected")
                if healthy_state is not None:
                    model.load_state_dict(healthy_state)
                    stop_reason += " -> ROLLED BACK to last healthy checkpoint"
                break

            # retain is healthy at this point — snapshot for possible rollback
            healthy_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if forget_now >= FORGET_LOSS_TARGET:
                stop_reason = f"forget-loss hit target ({forget_now:.2f} ≥ {FORGET_LOSS_TARGET})"
                break

    final_forget = eval_loss(model, forget_loader)
    final_retain = eval_loss(model, retain_loader)
    print(f"\n[Phase 4] Stopped: {stop_reason}")
    print(f"[Phase 4] Final losses — forget: {base_forget:.3f} -> {final_forget:.3f}, "
          f"retain: {base_retain:.3f} -> {final_retain:.3f}")
    return model, {"forget_loss": (base_forget, final_forget),
                   "retain_loss": (base_retain, final_retain),
                   "stop_reason": stop_reason}


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(int(os.environ.get("SEED", 42)))   # reproducible batch order
    ckpt_dir = get_checkpoint_dir()
    trained_path = os.path.join(ckpt_dir, "trained_model.pt")
    assert os.path.exists(trained_path), \
        f"Run train.py first — no checkpoint at {trained_path}"

    tokenizer = get_tokenizer()
    loaders, datasets = make_dataloaders(tokenizer)

    print(f"[Phase 4] Loading trained checkpoint: {trained_path}")
    model = load_checkpoint(trained_path)

    # quick BEFORE sample so the change is visible immediately
    probe = datasets["forget"].examples[0]
    print(f"\n[Phase 4] BEFORE unlearning — Q: {probe['question']}")
    print(f"[Phase 4]   model: {ask_model(model, tokenizer, probe['question'])[:150]}")

    model, stats = unlearn(model, loaders["forget"], loaders["retain"])

    print(f"\n[Phase 4] AFTER unlearning — same question:")
    print(f"[Phase 4]   model: {ask_model(model, tokenizer, probe['question'])[:150]}")

    unlearned_path = os.path.join(ckpt_dir, "unlearned_model.pt")
    save_checkpoint(model, unlearned_path, extra={"phase": "unlearned", **stats})

    print("\n[Phase 4] ✅ Unlearning complete. Both checkpoints saved:")
    print(f"           trained   : {trained_path}")
    print(f"           unlearned : {unlearned_path}")
    print("           Next: Phase 5 (audit — was the forgetting real?).")
