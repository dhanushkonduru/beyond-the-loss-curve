"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 8 — Sequential Unlearning (sustainability criterion)
=============================================================

Real deletion requests arrive one after another, not all at once.
This experiment unlearns the 20 forget countries in 4 successive
ROUNDS of 5 countries each, measuring model health after every round.

MUSE found most methods degrade catastrophically under sequential
requests — this tests whether GA does too, at our scale.

Writes results/sequential_report.txt.  Run:  python sequential.py
"""

import copy
import os

import torch
from torch.utils.data import DataLoader

from model import DEVICE
from data import QA_PER_ENTITY, collate, get_tokenizer, make_dataloaders
from unlearn_ga import eval_loss, unlearn
from train import get_checkpoint_dir, load_checkpoint

AUTHORS_PER_ROUND = 5
ROUNDS = 4
STEPS_PER_ROUND = 50   # gentler per-round budget than one-shot unlearning

if __name__ == "__main__":
    torch.manual_seed(int(os.environ.get("SEED", 42)))
    ckpt_dir = get_checkpoint_dir()
    tokenizer = get_tokenizer()
    loaders, datasets = make_dataloaders(tokenizer)

    trained_path = os.path.join(ckpt_dir, "trained_model.pt")
    model = load_checkpoint(trained_path)

    forget_ds = datasets["forget"]
    rows_per_round = AUTHORS_PER_ROUND * QA_PER_ENTITY
    collate_fn = lambda b: collate(b, tokenizer.pad_token_id)

    def subset_loader(lo, hi):
        ds = copy.copy(forget_ds)
        ds.examples = forget_ds.examples[lo:hi]
        return DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

    base_retain = eval_loss(model, loaders["retain"])
    lines = ["FORGET-ME-NOT — SEQUENTIAL UNLEARNING (4 rounds x 5 patients)", "=" * 70,
             f"start: retain-loss {base_retain:.3f}", "",
             f"{'round':>5} {'patients':>10} {'loss(all forgotten so far)':>28} {'retain-loss':>12} {'drift':>8}"]

    for r in range(ROUNDS):
        lo, hi = r * rows_per_round, (r + 1) * rows_per_round
        round_loader = subset_loader(lo, hi)
        print(f"\n[Phase 8] ===== ROUND {r + 1}/{ROUNDS}: patients {r * 5 + 1}-{r * 5 + 5} =====")
        model, _ = unlearn(model, round_loader, loaders["retain"], max_steps=STEPS_PER_ROUND)

        cumulative_loader = subset_loader(0, hi)
        f_loss = eval_loss(model, cumulative_loader)
        r_loss = eval_loss(model, loaders["retain"])
        lines.append(f"{r + 1:>5} {f'{lo//20 + 1}-{hi//20}':>10} {f_loss:>28.3f} {r_loss:>12.3f} "
                     f"{r_loss - base_retain:>+8.3f}")

    lines += ["", "Reading: if retain-drift grows round over round, the method does not",
              "sustain repeated deletion requests (each request damages the model further)."]
    report = "\n".join(lines)
    print("\n" + report)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sequential_report.txt")
    with open(out, "w") as f:
        f.write(report)
    torch.save({"model_state": model.state_dict(), "preset": "small-50m",
                "phase": "sequential_unlearned"},
               os.path.join(ckpt_dir, "sequential_model.pt"))
    print(f"\n[Phase 8] ✅ Written -> {out}")
