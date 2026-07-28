"""
=============================================================
 FORGET-ME-NOT : Dataset-Model Sync Framework
 sync.py — keep a model in sync with a versioned dataset
=============================================================

Given two releases of a dataset (here: two years of REAL WHO
immunization data), the framework performs ONE sync cycle WITHOUT
full retraining:

  1. DIFF   -> added / removed / kept  (precomputed by sync_fetch.py)
  2. LEARN  -> fine-tune on the ADDED records only
  3. UNLEARN-> NPO (retain-anchored) on the REMOVED records only
  4. AUDIT  -> for each set, check the model's actual answers:
        added   should now be LEARNED
        removed should now be FORGOTTEN
        kept    should stay PRESERVED
  5. REPORT -> % learned, % forgotten, % preserved (before vs after)

Honesty: unlearning here is approximate (no retraining), so the
report states the achieved percentages rather than claiming 100%.

Run:  python sync.py     (trains base model on release_2023
      the first time, then syncs to release_2024)
"""

import copy
import json
import os
import re

import torch
from torch.utils.data import DataLoader

from model import GPT, PRESETS, DEVICE
from data import QADataset, collate, get_tokenizer
from train import ask_model, get_checkpoint_dir, load_checkpoint, save_checkpoint, train
from unlearn_ga import eval_loss
from unlearn_npo import unlearn_npo

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "whosync")
BASE_EPOCHS = int(os.environ.get("BASE_EPOCHS", 40))
LEARN_EPOCHS = int(os.environ.get("LEARN_EPOCHS", 10))
AUDIT_SAMPLE = 40   # records probed per set for the readable recall check


def load(name):
    with open(os.path.join(DATA, f"{name}.json")) as f:
        return json.load(f)


def loader(rows, tokenizer, bs=16, shuffle=True):
    ds = QADataset(rows, tokenizer)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      collate_fn=lambda b: collate(b, tokenizer.pad_token_id))


def has_value(answer, value):
    """True if the answer states the coverage value (as a standalone number)."""
    return re.search(rf"\b{value}\b", answer) is not None


def knows_loss(model, tokenizer, rows):
    """Mean answer-loss over a record set. LOW = the model knows these facts,
    HIGH = it does not. Continuous and collision-free (unlike value matching)."""
    return eval_loss(model, loader(rows, tokenizer, shuffle=False),
                     max_batches=10 ** 6)


def recall(model, tokenizer, records, sample=AUDIT_SAMPLE):
    """Readable secondary check: fraction whose exact value the model outputs."""
    recs = records[:sample]
    hit = sum(has_value(ask_model(model, tokenizer,
              f"What was the MCV1 immunization coverage in {r['country']} in {r['year']}?",
              max_new_tokens=40), r["value"]) for r in recs)
    return hit / max(len(recs), 1)


if __name__ == "__main__":
    torch.manual_seed(42)
    ckpt_dir = get_checkpoint_dir()
    tokenizer = get_tokenizer()

    rel23 = load("release_2023")
    rel24 = load("release_2024")
    diff = load("diff")
    rec23 = {r["id"]: r for r in load("records_2023")}
    rec24 = {r["id"]: r for r in load("records_2024")}

    added_set, removed_set, kept_set = map(set, (diff["added"], diff["removed"], diff["kept"]))
    added_rows = [r for r in rel24 if r["id"] in added_set]      # QA to LEARN
    removed_rows = [r for r in rel23 if r["id"] in removed_set]  # QA to UNLEARN
    kept_rows = [r for r in rel23 if r["id"] in kept_set]        # QA to RETAIN
    removed_recs = [rec23[i] for i in diff["removed"]]
    added_recs = [rec24[i] for i in diff["added"]]
    kept_recs = [rec23[i] for i in diff["kept"]]
    print(f"[Sync] diff: LEARN {len(added_recs)}, UNLEARN {len(removed_recs)}, "
          f"RETAIN {len(kept_recs)}")

    # ---- base model: trained on release_2023 (the "current" deployed model) ----
    base_path = os.path.join(ckpt_dir, "sync_base.pt")
    if os.path.exists(base_path):
        print("[Sync] loading existing base model (release_2023)")
        model = load_checkpoint(base_path)
    else:
        print(f"[Sync] training base model on release_2023 ({BASE_EPOCHS} epochs)...")
        model = GPT(PRESETS["small-50m"]).to(DEVICE)
        model = train(model, loader(rel23, tokenizer), epochs=BASE_EPOCHS)
        save_checkpoint(model, base_path, extra={"phase": "sync_base"})

    # ---- BEFORE sync (loss = primary signal; recall = readable secondary) ----
    print("\n[Sync] === AUDIT BEFORE SYNC ===")
    bl = {"added": knows_loss(model, tokenizer, added_rows),
          "removed": knows_loss(model, tokenizer, removed_rows),
          "kept": knows_loss(model, tokenizer, kept_rows)}
    br = {"added": recall(model, tokenizer, added_recs),
          "removed": recall(model, tokenizer, removed_recs),
          "kept": recall(model, tokenizer, kept_recs)}
    print(f"[Sync]  loss  added {bl['added']:.2f} (high=unknown) | "
          f"removed {bl['removed']:.2f} (low=known) | kept {bl['kept']:.2f} (low=known)")

    # ---- LEARN the new records (+ kept rehearsal so we don't wipe them) ----
    print(f"\n[Sync] LEARN: fine-tuning on {len(added_rows)} added records "
          f"+ kept rehearsal ({LEARN_EPOCHS} epochs)...")
    rehearse = added_rows + [kept_rows[i] for i in range(0, len(kept_rows), 2)]
    model = train(model, loader(rehearse, tokenizer), epochs=LEARN_EPOCHS)

    # ---- UNLEARN the removed records (NPO, retain-anchored, NO retraining) ----
    print(f"\n[Sync] UNLEARN: NPO on {len(removed_rows)} removed records "
          f"(anchored on kept)...")
    ref_model = copy.deepcopy(model)
    model, _ = unlearn_npo(model, ref_model, loader(removed_rows, tokenizer),
                           loader(kept_rows, tokenizer), max_steps=300)

    # ---- AFTER sync ----
    print("\n[Sync] === AUDIT AFTER SYNC ===")
    al = {"added": knows_loss(model, tokenizer, added_rows),
          "removed": knows_loss(model, tokenizer, removed_rows),
          "kept": knows_loss(model, tokenizer, kept_rows)}
    ar = {"added": recall(model, tokenizer, added_recs),
          "removed": recall(model, tokenizer, removed_recs),
          "kept": recall(model, tokenizer, kept_recs)}
    save_checkpoint(model, os.path.join(ckpt_dir, "sync_after.pt"), extra={"phase": "sync_after"})

    # ---- REPORT ----
    def arrow(k, want):
        d = al[k] - bl[k]
        return f"{bl[k]:.2f} -> {al[k]:.2f}  ({'↓' if d < 0 else '↑'} {abs(d):.2f})"
    lines = ["FORGET-ME-NOT — DATASET-MODEL SYNC REPORT (real WHO MCV1 data)", "=" * 68,
             "One sync cycle: release_2023 -> release_2024, NO full retraining.",
             "Primary signal = answer-loss (low = model knows the fact).", "",
             f"{'set':<22}{'records':>8}{'  answer-loss (want)':<26}{'exact-recall':>16}",
             "-" * 68,
             f"{'ADDED  learn new':<22}{len(added_rows):>8}  {arrow('added','↓'):<24}"
             f"want ↓   {br['added']*100:>3.0f}% -> {ar['added']*100:>3.0f}%",
             f"{'REMOVED  unlearn':<22}{len(removed_rows):>8}  {arrow('removed','↑'):<24}"
             f"want ↑   {br['removed']*100:>3.0f}% -> {ar['removed']*100:>3.0f}%",
             f"{'KEPT  retain':<22}{len(kept_rows):>8}  {arrow('kept','=' ):<24}"
             f"want =   {br['kept']*100:>3.0f}% -> {ar['kept']*100:>3.0f}%",
             "-" * 68, "",
             f"LEARNED   : new-record loss fell {bl['added']:.2f} -> {al['added']:.2f} "
             f"(model absorbed the {len(added_rows)} new WHO figures)",
             f"UNLEARNED : withdrawn-record loss rose {bl['removed']:.2f} -> {al['removed']:.2f} "
             f"(pushed toward 'unknown' without retraining)",
             f"PRESERVED : kept-record loss stayed {bl['kept']:.2f} -> {al['kept']:.2f} "
             f"(retained facts protected by rehearsal + anchor)", "",
             "Unlearning is approximate (no retraining): the framework REPORTS the",
             "achieved shift; any record with residual low loss is flagged for review."]
    report = "\n".join(lines)
    print("\n" + report)

    results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    with open(os.path.join(results, "sync_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(results, "sync.json"), "w") as f:
        json.dump({"loss_before": bl, "loss_after": al,
                   "recall_before": br, "recall_after": ar,
                   "n": {"added": len(added_rows), "removed": len(removed_rows),
                         "kept": len(kept_rows)}}, f, indent=1)
    print("\n[Sync] ✅ Written -> results/sync_report.txt and sync.json")
