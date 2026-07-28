"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 4c — Unlearning via NPO (Negative Preference Optimization)
=============================================================

Gradient Ascent diverges because -loss is unbounded: it keeps pushing
forever and destroys the model. NPO (Zhang et al., 2024) fixes this
with a DPO-style bounded objective that treats the forget answers as a
"dispreferred" response relative to a frozen reference model:

    L_forget = (2/beta) * log(1 + (p_theta/p_ref)^beta)

As the current model's likelihood on the forget data drops below the
reference's, the gradient smoothly SATURATES instead of exploding —
so NPO forgets hard without catastrophic collapse. We add the standard
retain term (like gradient difference) to protect utility:

    L = L_npo(forget) + retain_weight * loss(retain)

This is the current strong baseline for LLM unlearning. Same audit,
same trust score — it should beat both GA and GD.

Requires: checkpoints/trained_model.pt.  Run:  python unlearn_npo.py
Saves: checkpoints/unlearned_npo.pt
"""

import os

import torch
import torch.nn.functional as F

from model import DEVICE
from data import get_tokenizer, make_dataloaders
from train import ask_model, get_checkpoint_dir, load_checkpoint, save_checkpoint
from unlearn_ga import eval_loss, forever

NPO_LR = 5e-5
NPO_BETA = 0.1           # DPO temperature; smaller = gentler saturation
RETAIN_WEIGHT = 1.0
MAX_STEPS = 400
GRAD_CLIP = 1.0
LOG_EVERY = 10
FORGET_LOSS_TARGET = 3.0
RETAIN_DRIFT_LIMIT = 2.0


def seq_logprob(model, batch):
    """Mean per-token log-prob of the answer tokens (labels != -100)."""
    ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    logits, _ = model(ids)
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    tgt_safe = tgt.clamp(min=0)
    tok = logp.gather(-1, tgt_safe.unsqueeze(-1)).squeeze(-1)
    tok = tok * mask
    return tok.sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def unlearn_npo(model, ref_model, forget_loader, retain_loader, max_steps=MAX_STEPS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=NPO_LR)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    base_forget = eval_loss(model, forget_loader)
    base_retain = eval_loss(model, retain_loader)
    print(f"[Phase 4c] Baseline — forget: {base_forget:.3f}, retain: {base_retain:.3f}")
    print(f"[Phase 4c] NPO (beta={NPO_BETA}) with retain anchor; target forget-loss {FORGET_LOSS_TARGET}")

    model.train()
    fbatches, rbatches = forever(forget_loader), forever(retain_loader)
    stop_reason = f"reached step cap ({max_steps})"
    healthy_state = None

    for step in range(1, max_steps + 1):
        fb, rb = next(fbatches), next(rbatches)
        optimizer.zero_grad(set_to_none=True)

        lp = seq_logprob(model, fb)
        with torch.no_grad():
            lp_ref = seq_logprob(ref_model, fb)
        # NPO forget loss: bounded, saturates as lp falls below lp_ref
        npo = (2.0 / NPO_BETA) * F.softplus(NPO_BETA * (lp - lp_ref)).mean()

        _, r_loss = model(rb["input_ids"].to(DEVICE), targets=rb["labels"].to(DEVICE))
        (npo + RETAIN_WEIGHT * r_loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            forget_now = eval_loss(model, forget_loader)
            retain_now = eval_loss(model, retain_loader)
            print(f"[Phase 4c]   step {step:>3}/{max_steps} | "
                  f"forget-loss {forget_now:.3f} (▲ from {base_forget:.3f}) | "
                  f"retain-loss {retain_now:.3f} (baseline {base_retain:.3f})")
            if retain_now - base_retain > RETAIN_DRIFT_LIMIT:
                stop_reason = f"retain-loss drifted {retain_now - base_retain:.2f} above baseline"
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
    print(f"\n[Phase 4c] Stopped: {stop_reason}")
    print(f"[Phase 4c] Final — forget: {base_forget:.3f} -> {final_forget:.3f}, "
          f"retain: {base_retain:.3f} -> {final_retain:.3f}")
    return model, {"forget_loss": (base_forget, final_forget),
                   "retain_loss": (base_retain, final_retain), "stop_reason": stop_reason}


if __name__ == "__main__":
    torch.manual_seed(int(os.environ.get("SEED", 42)))
    ckpt_dir = get_checkpoint_dir()
    trained_path = os.path.join(ckpt_dir, "trained_model.pt")
    assert os.path.exists(trained_path), "Run train.py first."

    tokenizer = get_tokenizer()
    loaders, datasets = make_dataloaders(tokenizer)
    model = load_checkpoint(trained_path)
    ref_model = load_checkpoint(trained_path)   # frozen reference

    probe = datasets["forget"].examples[0]
    print(f"\n[Phase 4c] BEFORE — Q: {probe['question']}")
    print(f"[Phase 4c]   model: {ask_model(model, tokenizer, probe['question'])[:150]}")

    model, stats = unlearn_npo(model, ref_model, loaders["forget"], loaders["retain"])

    print(f"\n[Phase 4c] AFTER — same question:")
    print(f"[Phase 4c]   model: {ask_model(model, tokenizer, probe['question'])[:150]}")

    save_checkpoint(model, os.path.join(ckpt_dir, "unlearned_npo.pt"),
                    extra={"phase": "unlearned_npo", **stats})
    print("\n[Phase 4c] ✅ Saved -> checkpoints/unlearned_npo.pt")
