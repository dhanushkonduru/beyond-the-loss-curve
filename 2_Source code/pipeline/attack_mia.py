"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 7 — Membership Inference Attack (privacy criterion)
=============================================================

Question: can an attacker tell that the forget countries' data WAS
in training, just by looking at the model's losses?

Method (loss-based MIA, MUSE "PrivLeak" style):
  members     = the 400 forget QA pairs (were trained on)
  non-members = the 400 held-out QA pairs (20 real WHO countries
                the model NEVER trained on — genuine non-members)
  For each model, compute per-example loss on both groups and the
  AUC of separating members from non-members by "lower loss".

Reading the result:
  AUC ~ 1.0 : members have clearly lower loss -> data footprint intact
  AUC ~ 0.5 : indistinguishable -> no privacy leakage
  The ORACLE's AUC is the reference for "naturally ~0.5" — an
  unlearned model should match the oracle, not undershoot it.

Writes results/mia_report.txt.  Run:  python attack_mia.py
"""

import json
import os

import torch

from model import DEVICE
from data import DATA_DIR, get_tokenizer
from train import get_checkpoint_dir, load_checkpoint

MODELS = [
    ("trained (before unlearning)", "trained_model.pt"),
    ("Gradient Ascent", "unlearned_model.pt"),
    ("Gradient Difference", "unlearned_gd.pt"),
    ("NPO", "unlearned_npo.pt"),
    ("ORACLE (retrained w/o forget)", "oracle_model.pt"),
]


@torch.no_grad()
def per_example_losses(model, rows, tokenizer, qfield="question", afield="answer"):
    """Answer-token loss for each QA pair individually."""
    losses = []
    eos = tokenizer.eos_token_id
    for row in rows:
        q, a = row[qfield], row[afield]
        if isinstance(a, list):
            a = a[0]
        prompt_ids = tokenizer(f"Question: {q} Answer:").input_ids
        full_ids = tokenizer(f"Question: {q} Answer: {a}").input_ids + [eos]
        full_ids = full_ids[:256]
        labels = list(full_ids)
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels[:n_prompt] = [-100] * n_prompt
        ids_t = torch.tensor([full_ids], device=DEVICE)
        lab_t = torch.tensor([labels], device=DEVICE)
        _, loss = model(ids_t, targets=lab_t)
        losses.append(loss.item())
    return losses


def auc_lower_loss_is_member(member_losses, nonmember_losses):
    """Rank-based AUC: probability a random member has LOWER loss
    than a random non-member (Mann-Whitney, no scipy needed)."""
    wins, ties, total = 0, 0, len(member_losses) * len(nonmember_losses)
    for m in member_losses:
        for n in nonmember_losses:
            if m < n:
                wins += 1
            elif m == n:
                ties += 1
    return (wins + 0.5 * ties) / total


if __name__ == "__main__":
    ckpt_dir = get_checkpoint_dir()
    tokenizer = get_tokenizer()

    print("[Phase 7] Loading forget (members) and held-out countries (non-members)...")
    with open(os.path.join(DATA_DIR, "forget.json")) as f:
        members = json.load(f)
    with open(os.path.join(DATA_DIR, "heldout.json")) as f:
        nonmembers = json.load(f)
    print(f"[Phase 7]   members: {len(members)}, non-members: {len(nonmembers)}")

    lines = ["FORGET-ME-NOT — MEMBERSHIP INFERENCE ATTACK (loss-based)", "=" * 70,
             f"members = forget countries ({len(members)}), "
             f"non-members = held-out countries never trained on ({len(nonmembers)})", "",
             f"{'model':<32} {'AUC':>8}   interpretation"]
    aucs = {}
    for name, fname in MODELS:
        path = os.path.join(ckpt_dir, fname)
        if not os.path.exists(path):
            print(f"[Phase 7] SKIP {name} (missing checkpoint)")
            continue
        print(f"[Phase 7] Attacking: {name}")
        model = load_checkpoint(path)
        model.eval()
        ml = per_example_losses(model, members, tokenizer)
        # non-members: real WHO countries the model never trained on
        nl = per_example_losses(model, nonmembers, tokenizer)
        auc = auc_lower_loss_is_member(ml, nl)
        aucs[name] = auc
        verdict = ("no leakage" if abs(auc - 0.5) < 0.1
                   else "LEAKING membership" if auc > 0.5
                   else "over-unlearned (inverted signal — also a leak)")
        lines.append(f"{name:<32} {auc:>8.3f}   {verdict}")
        del model

    oracle_auc = aucs.get("ORACLE (retrained w/o forget)")
    if oracle_auc is not None:
        lines += ["", f"Oracle reference AUC: {oracle_auc:.3f} "
                      "(the 'natural' value an unlearned model should match)"]
        for name, auc in aucs.items():
            if not name.startswith("ORACLE"):
                lines.append(f"  PrivLeak({name}) = AUC - oracle = {auc - oracle_auc:+.3f}")

    report = "\n".join(lines)
    print("\n" + report)
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    with open(os.path.join(results_dir, "mia_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(results_dir, "mia.json"), "w") as f:
        json.dump(aucs, f, indent=1)
    print(f"\n[Phase 7] ✅ Written -> results/mia_report.txt and mia.json")
