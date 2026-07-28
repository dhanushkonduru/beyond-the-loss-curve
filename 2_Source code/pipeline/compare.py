"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 6b — Comparative Analysis (thesis Chapter 6.2)
=============================================================

Puts every checkpoint side by side against the retrain-from-scratch
ORACLE — the gold standard the literature demands:

  trained          : knows everything (upper bound of memorization)
  unlearned_safe   : conservative GA (retain protected)
  unlearned_model  : aggressive GA (forgetting prioritized)
  oracle           : never saw the forget authors (the target behavior)

For each model: loss on forget set, loss on retain set, and the gap
from oracle behavior. An ideal unlearned model matches the oracle on
BOTH numbers.

Requires all checkpoints. Run:  python compare.py
Writes results/comparison.txt
"""

import os

from model import DEVICE
from data import get_tokenizer, make_dataloaders
from train import ask_model, get_checkpoint_dir, load_checkpoint
from unlearn_ga import eval_loss

MODELS = [
    ("trained (before unlearning)", "trained_model.pt"),
    ("Gradient Ascent", "unlearned_model.pt"),
    ("Gradient Difference", "unlearned_gd.pt"),
    ("NPO", "unlearned_npo.pt"),
    ("ORACLE (retrained w/o forget)", "oracle_model.pt"),
]

if __name__ == "__main__":
    ckpt_dir = get_checkpoint_dir()
    tokenizer = get_tokenizer()
    loaders, datasets = make_dataloaders(tokenizer)

    rows = []
    probe = datasets["forget"].examples[0]
    answers = []
    for name, fname in MODELS:
        path = os.path.join(ckpt_dir, fname)
        if not os.path.exists(path):
            print(f"[Phase 6b] SKIPPING {name} — missing {path}")
            continue
        print(f"[Phase 6b] Evaluating: {name}")
        model = load_checkpoint(path)
        model.eval()
        fl = eval_loss(model, loaders["forget"], max_batches=25)
        rl = eval_loss(model, loaders["retain"], max_batches=50)
        rows.append((name, fl, rl))
        answers.append((name, ask_model(model, tokenizer, probe["question"])[:110]))
        del model

    oracle = next((r for r in rows if r[0].startswith("ORACLE")), None)

    lines = ["FORGET-ME-NOT — COMPARATIVE ANALYSIS (all models vs oracle)", "=" * 78,
             f"{'model':<32} {'forget-loss':>12} {'retain-loss':>12} {'|Δforget|':>10} {'|Δretain|':>10}",
             "-" * 78]
    for name, fl, rl in rows:
        if oracle:
            df, dr = abs(fl - oracle[1]), abs(rl - oracle[2])
            lines.append(f"{name:<32} {fl:>12.3f} {rl:>12.3f} {df:>10.3f} {dr:>10.3f}")
        else:
            lines.append(f"{name:<32} {fl:>12.3f} {rl:>12.3f} {'—':>10} {'—':>10}")
    lines += ["-" * 78,
              "Δ columns: distance from oracle behavior (smaller = closer to true unlearning).",
              "Note: an unlearned model should MATCH the oracle's forget-loss, not exceed it —",
              "overshooting means confidently-wrong answers, itself an information leak (MUSE).",
              "", f"Sample question: {probe['question']}"]
    lines += [f"  {name:<30}: {ans}" for name, ans in answers]

    report = "\n".join(lines)
    print("\n" + report)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "comparison.txt")
    with open(out, "w") as f:
        f.write(report)
    print(f"\n[Phase 6b] ✅ Written -> {out}")
