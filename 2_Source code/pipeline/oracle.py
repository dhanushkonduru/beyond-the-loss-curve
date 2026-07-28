"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 6a — Retrain-from-Scratch ORACLE
=============================================================

The gold standard of unlearning: a model trained on the retain
countries ONLY, which has genuinely NEVER seen the 20 forget countries.

Every unlearning method should be compared against this oracle
(not against zero!): a properly unlearned model should behave
like the oracle on forget data — clueless, but not scrambled.

Used in Chapter 6 (comparative analysis).
Run:  EPOCHS=20 python oracle.py
"""

import os

from model import GPT, PRESETS, DEVICE
from data import get_tokenizer, make_dataloaders
from train import ask_model, get_checkpoint_dir, save_checkpoint, train
from unlearn_ga import eval_loss

EPOCHS = int(os.environ.get("EPOCHS", 20))

if __name__ == "__main__":
    import torch
    torch.manual_seed(int(os.environ.get("SEED", 42)))   # reproducible runs
    ckpt_dir = get_checkpoint_dir()
    tokenizer = get_tokenizer()
    loaders, datasets = make_dataloaders(tokenizer)

    print(f"[Phase 6a] Training ORACLE on retain90 only ({EPOCHS} epochs, "
          f"{len(loaders['retain'])} batches/epoch)...")
    model = GPT(PRESETS["small-50m"]).to(DEVICE)
    model = train(model, loaders["retain"], epochs=EPOCHS)

    forget_loss = eval_loss(model, loaders["forget"])
    retain_loss = eval_loss(model, loaders["retain"])
    print(f"\n[Phase 6a] Oracle losses — forget: {forget_loss:.3f} (never seen these authors), "
          f"retain: {retain_loss:.3f}")
    print("[Phase 6a] These are the REFERENCE values: an ideal unlearned model "
          "should match them.")

    save_checkpoint(model, os.path.join(ckpt_dir, "oracle_model.pt"),
                    extra={"phase": "oracle", "epochs": EPOCHS,
                           "forget_loss": forget_loss, "retain_loss": retain_loss})

    # what does "never knew them" look like? (contrast with scrambled GA output)
    probe = datasets["forget"].examples[0]
    print(f"\n[Phase 6a] Oracle asked about a forget author it never saw:")
    print(f"  Q: {probe['question']}")
    print(f"  oracle: {ask_model(model, tokenizer, probe['question'])[:150]}")
    print("\n[Phase 6a] ✅ Oracle saved -> checkpoints/oracle_model.pt")
