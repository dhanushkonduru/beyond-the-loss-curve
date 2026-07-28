"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 10 — Operating-Point Sweep (GA vs GD frontier)
=============================================================

A single tuned stopping point invites "you cherry-picked it." This
sweeps BOTH methods across a range of unlearning budgets (retain-drift
limits) and records the forgetting/utility each reaches — tracing the
whole trade-off curve, not one dot.

If GD's curve sits above/left of GA's at every budget, GD *dominates*
GA — a far stronger claim than one comparison.

Writes results/sweep_report.txt and sweep.json (for the figure).
Run:  python sweep.py
"""

import copy
import json
import os

import torch

from model import DEVICE
from data import get_tokenizer, make_dataloaders
from train import get_checkpoint_dir, load_checkpoint
from unlearn_ga import eval_loss, forever

DRIFT_LIMITS = [0.03, 0.06, 0.12, 0.25, 0.5]   # unlearning "budgets"
LR = 1e-5
GD_LAMBDA = 0.4
MAX_STEPS = 400
LOG = 10


def run(method, forget_loader, retain_loader, drift_limit, base_forget, base_retain):
    """Run GA or GD until retain-loss drifts past `drift_limit`; return the
    forget-loss and retain-loss reached at that healthy operating point."""
    model = load_checkpoint(os.path.join(get_checkpoint_dir(), "trained_model.pt"))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    fb, rb = forever(forget_loader), forever(retain_loader)
    model.train()
    healthy = (base_forget, base_retain)
    for step in range(1, MAX_STEPS + 1):
        f = next(fb)
        opt.zero_grad(set_to_none=True)
        _, f_loss = model(f["input_ids"].to(DEVICE), targets=f["labels"].to(DEVICE))
        if method == "GA":
            (-f_loss).backward()
        else:  # GD
            r = next(rb)
            _, r_loss = model(r["input_ids"].to(DEVICE), targets=r["labels"].to(DEVICE))
            (-f_loss + GD_LAMBDA * r_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % LOG == 0:
            fn = eval_loss(model, forget_loader)
            rn = eval_loss(model, retain_loader)
            if rn - base_retain > drift_limit:
                break
            healthy = (fn, rn)
    del model
    return healthy


if __name__ == "__main__":
    torch.manual_seed(int(os.environ.get("SEED", 42)))
    tokenizer = get_tokenizer()
    loaders, _ = make_dataloaders(tokenizer)
    m0 = load_checkpoint(os.path.join(get_checkpoint_dir(), "trained_model.pt"))
    base_forget = eval_loss(m0, loaders["forget"])
    base_retain = eval_loss(m0, loaders["retain"])
    del m0
    print(f"[Phase 10] baseline forget {base_forget:.3f} retain {base_retain:.3f}")

    data = {"GA": [], "GD": [], "base": [base_forget, base_retain]}
    for method in ("GA", "GD"):
        for dl in DRIFT_LIMITS:
            print(f"[Phase 10] {method} @ drift≤{dl}")
            f, r = run(method, loaders["forget"], loaders["retain"], dl, base_forget, base_retain)
            data[method].append({"drift_limit": dl,
                                 "forget_rise": f - base_forget,
                                 "retain_drift": r - base_retain})

    lines = ["FORGET-ME-NOT — OPERATING-POINT SWEEP (GA vs GD)", "=" * 70,
             f"baseline forget-loss {base_forget:.3f}, retain-loss {base_retain:.3f}", "",
             f"{'method':<6} {'budget':>8} {'forget-rise':>13} {'retain-drift':>14}"]
    for method in ("GA", "GD"):
        for pt in data[method]:
            lines.append(f"{method:<6} {pt['drift_limit']:>8} {pt['forget_rise']:>13.3f} "
                         f"{pt['retain_drift']:>14.3f}")
    lines += ["", "Higher forget-rise at the same retain-drift = better method.",
              "If GD's forget-rise exceeds GA's at every budget, GD dominates."]
    report = "\n".join(lines)
    print("\n" + report)
    results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    with open(os.path.join(results, "sweep_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(results, "sweep.json"), "w") as f:
        json.dump(data, f, indent=1)
    print("\n[Phase 10] ✅ Written -> results/sweep_report.txt and sweep.json")
