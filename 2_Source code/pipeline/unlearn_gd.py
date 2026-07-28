"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 4b — Unlearning via GRADIENT DIFFERENCE
=============================================================

Gradient Ascent's fatal flaw: it only pushes AWAY from forget data,
and everything nearby gets dragged along (retain damage in lockstep).

Gradient Difference fixes this with a two-sided objective per step:

    loss = - loss(forget batch) + LAMBDA * loss(retain batch)
            ^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^^
            push away from        while actively holding on
            forgotten patients    to everyone else

The retain term acts as an anchor, so we can push forgetting much
further before utility degrades — a genuinely better method that the
SAME audit can score against GA.

Requires: checkpoints/trained_model.pt.  Run:  python unlearn_gd.py
Saves: checkpoints/unlearned_gd.pt
"""

import os

import torch

from model import DEVICE
from data import get_tokenizer, make_dataloaders
from train import ask_model, get_checkpoint_dir, load_checkpoint, save_checkpoint
from unlearn_ga import eval_loss, forever

# -------------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------------
GD_LR = 5e-5              # WHO facts memorize to ~0.000, so a bigger push is needed
GD_LAMBDA = 0.4           # retain-anchor weight — kept < 1 so the forget-push
                          # is not overwhelmed when data is templated (forget and
                          # retain look structurally alike, so a strong anchor
                          # would drag forget examples down with retain)
MAX_STEPS = 400           # anchor allows a longer, deeper run than GA
GRAD_CLIP = 1.0
LOG_EVERY = 10

FORGET_LOSS_TARGET = 3.0  # meaningful forgetting relative to ~0.000 baseline
RETAIN_DRIFT_LIMIT = 2.0  # WHO QA are heavily templated, so forget and retain rise
                          # together; a loose limit lets GD reach the forget target and
                          # the audit then measures the (real) utility cost of doing so
COLLAPSE_FLOOR = 0.002    # both losses sinking here = degenerate over-memorization


def unlearn_gd(model, forget_loader, retain_loader, max_steps=MAX_STEPS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=GD_LR)

    base_forget = eval_loss(model, forget_loader)
    base_retain = eval_loss(model, retain_loader)
    print(f"[Phase 4b] Baseline — forget: {base_forget:.3f}, retain: {base_retain:.3f}")
    print(f"[Phase 4b] Goal: forget-loss -> {FORGET_LOSS_TARGET} (oracle ≈ 7.0), "
          f"retain-loss stays near {base_retain:.3f}")

    model.train()
    fbatches, rbatches = forever(forget_loader), forever(retain_loader)
    stop_reason = f"reached step cap ({max_steps})"
    healthy_state = None
    peaked = False           # forget-loss starts at ~0; only guard "collapse" after it rises

    for step in range(1, max_steps + 1):
        fb, rb = next(fbatches), next(rbatches)

        optimizer.zero_grad(set_to_none=True)
        _, f_loss = model(fb["input_ids"].to(DEVICE), targets=fb["labels"].to(DEVICE))
        _, r_loss = model(rb["input_ids"].to(DEVICE), targets=rb["labels"].to(DEVICE))

        # the gradient-difference objective: ascend on forget, descend on retain
        (-f_loss + GD_LAMBDA * r_loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            forget_now = eval_loss(model, forget_loader)
            retain_now = eval_loss(model, retain_loader)
            print(f"[Phase 4b]   step {step:>3}/{max_steps} | "
                  f"forget-loss {forget_now:.3f} (▲ from {base_forget:.3f}) | "
                  f"retain-loss {retain_now:.3f} (baseline {base_retain:.3f})")

            if retain_now - base_retain > RETAIN_DRIFT_LIMIT:
                stop_reason = (f"retain-loss drifted {retain_now - base_retain:.2f} "
                               f"above baseline")
                if healthy_state is not None:
                    model.load_state_dict(healthy_state)
                    stop_reason += " -> ROLLED BACK to last healthy checkpoint"
                break
            # degenerate mode: forget-loss rose then sank back toward 0 = the anchor
            # won and dragged forget down with it. Only meaningful AFTER it has risen.
            peaked = peaked or forget_now > 0.1
            if peaked and forget_now < COLLAPSE_FLOOR and retain_now < COLLAPSE_FLOOR:
                stop_reason = "forget-loss collapsed toward 0 (anchor over-memorized)"
                if healthy_state is not None:
                    model.load_state_dict(healthy_state)
                    stop_reason += " -> ROLLED BACK to last healthy checkpoint"
                break
            healthy_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if forget_now >= FORGET_LOSS_TARGET:
                stop_reason = f"forget-loss hit target ({forget_now:.2f} ≥ {FORGET_LOSS_TARGET})"
                break

    final_forget = eval_loss(model, forget_loader)
    final_retain = eval_loss(model, retain_loader)
    print(f"\n[Phase 4b] Stopped: {stop_reason}")
    print(f"[Phase 4b] Final — forget: {base_forget:.3f} -> {final_forget:.3f}, "
          f"retain: {base_retain:.3f} -> {final_retain:.3f}")
    return model, {"forget_loss": (base_forget, final_forget),
                   "retain_loss": (base_retain, final_retain),
                   "stop_reason": stop_reason}


if __name__ == "__main__":
    torch.manual_seed(int(os.environ.get("SEED", 42)))
    ckpt_dir = get_checkpoint_dir()
    trained_path = os.path.join(ckpt_dir, "trained_model.pt")
    assert os.path.exists(trained_path), "Run train.py first."

    tokenizer = get_tokenizer()
    loaders, datasets = make_dataloaders(tokenizer)
    model = load_checkpoint(trained_path)

    probe = datasets["forget"].examples[0]
    print(f"\n[Phase 4b] BEFORE — Q: {probe['question']}")
    print(f"[Phase 4b]   model: {ask_model(model, tokenizer, probe['question'])[:150]}")

    model, stats = unlearn_gd(model, loaders["forget"], loaders["retain"])

    print(f"\n[Phase 4b] AFTER — same question:")
    print(f"[Phase 4b]   model: {ask_model(model, tokenizer, probe['question'])[:150]}")

    save_checkpoint(model, os.path.join(ckpt_dir, "unlearned_gd.pt"),
                    extra={"phase": "unlearned_gd", **stats})
    print("\n[Phase 4b] ✅ Saved -> checkpoints/unlearned_gd.pt")
