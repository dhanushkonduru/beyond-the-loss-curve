"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 9 — Relearning Attack (the "killer test")
=============================================================

If knowledge was truly ERASED, re-teaching it should take about as
long as teaching a fresh model. If it was only HIDDEN, it snaps back
in a handful of steps.

Method: take each unlearned checkpoint, fine-tune it on a TINY slice
of the forget data (a few countries, a few gradient steps), and measure
how fast the forget-loss drops back toward the original memorized level.
The oracle (never saw the data) is the reference for "honest" relearning
speed.

  fast recovery (few steps)  -> knowledge was hidden, not erased
  slow recovery (like oracle) -> knowledge was genuinely gone

Writes results/relearn_report.txt and relearn.json (for the figure).
Run:  python attack_relearn.py
"""

import copy
import json
import os

import torch
from torch.utils.data import DataLoader

from model import DEVICE
from data import collate, get_tokenizer, make_dataloaders
from train import get_checkpoint_dir, load_checkpoint
from unlearn_ga import eval_loss

MODELS = [
    ("Gradient Ascent", "unlearned_model.pt"),
    ("Gradient Difference", "unlearned_gd.pt"),
    ("NPO", "unlearned_npo.pt"),
    ("Oracle (never saw data)", "oracle_model.pt"),
]
RELEARN_STEPS = 20
RELEARN_LR = 5e-5
N_ATTACK_PATIENTS = 3   # attacker only has a few of the forgotten records


def relearn_curve(ckpt_path, attack_loader, forget_loader):
    model = load_checkpoint(ckpt_path)
    opt = torch.optim.AdamW(model.parameters(), lr=RELEARN_LR)
    batches = list(attack_loader)
    curve = [eval_loss(model, forget_loader)]
    model.train()
    step = 0
    while step < RELEARN_STEPS:
        for b in batches:
            opt.zero_grad(set_to_none=True)
            _, loss = model(b["input_ids"].to(DEVICE), targets=b["labels"].to(DEVICE))
            loss.backward()
            opt.step()
            step += 1
            if step % 2 == 0:
                curve.append(eval_loss(model, forget_loader))
            if step >= RELEARN_STEPS:
                break
    return curve


if __name__ == "__main__":
    torch.manual_seed(int(os.environ.get("SEED", 42)))
    ckpt_dir = get_checkpoint_dir()
    tokenizer = get_tokenizer()
    loaders, datasets = make_dataloaders(tokenizer)

    # attacker holds only the first few forgotten patients (few records)
    attack_ds = copy.copy(datasets["forget"])
    attack_ds.examples = datasets["forget"].examples[:N_ATTACK_PATIENTS * 20]
    attack_loader = DataLoader(attack_ds, batch_size=16, shuffle=True,
                               collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    orig_forget = eval_loss(load_checkpoint(os.path.join(ckpt_dir, "trained_model.pt")),
                            loaders["forget"])
    lines = ["FORGET-ME-NOT — RELEARNING ATTACK", "=" * 70,
             f"Attacker fine-tunes on {N_ATTACK_PATIENTS} forgotten patients for "
             f"{RELEARN_STEPS} steps.",
             f"Original memorized forget-loss (trained model): {orig_forget:.3f}", "",
             f"{'model':<26} {'start':>8} {'after':>8} {'recovered %':>12}"]
    curves = {}
    for name, fname in MODELS:
        path = os.path.join(ckpt_dir, fname)
        if not os.path.exists(path):
            print(f"[Phase 9] SKIP {name} (missing {fname})")
            continue
        print(f"[Phase 9] Attacking: {name}")
        curve = relearn_curve(path, attack_loader, loaders["forget"])
        curves[name] = curve
        start, after = curve[0], curve[-1]
        # how much of the gap back to the memorized level was recovered
        recov = 100 * (start - after) / max(start - orig_forget, 1e-6)
        lines.append(f"{name:<26} {start:>8.3f} {after:>8.3f} {recov:>11.1f}%")

    lines += ["", "Fast, near-complete recovery = knowledge was hidden, not erased.",
              "A method whose data snaps back in a few steps did not truly forget."]
    report = "\n".join(lines)
    print("\n" + report)
    results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    with open(os.path.join(results, "relearn_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(results, "relearn.json"), "w") as f:
        json.dump({"orig_forget": orig_forget, "steps_per_point": 2, "curves": curves}, f, indent=1)
    print(f"\n[Phase 9] ✅ Written -> results/relearn_report.txt and relearn.json")
