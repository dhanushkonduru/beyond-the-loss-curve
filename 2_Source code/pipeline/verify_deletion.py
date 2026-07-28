"""
=============================================================
 FORGET-ME-NOT : behavioural proof of deletion
 verify_deletion.py - the shortest demonstration of the paper's
 central claim, sized for a screen recording.

 Asks one deleted record and one retained record of BOTH the
 trained model and the unlearned model, then reports what each
 actually said. No loss curves: the evidence is the answer.

 Run:  python verify_deletion.py
 Takes about twenty seconds. Requires checkpoints/trained_model.pt
 and checkpoints/unlearned_npo.pt (produced by run_pipeline.sh).
=============================================================
"""

import json
import os
import re
import sys

from model import DEVICE
from data import get_tokenizer
from train import ask_model, get_checkpoint_dir, load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
W = 66


def rule(ch="="):
    print(ch * W)


def value_of(text):
    """The stated figure is the LAST number: earlier digits belong to the
    indicator name and the year (MCV1, DTP3, 2025)."""
    nums = re.findall(r"\d+", text)
    return int(nums[-1]) if nums else None


def load(name):
    with open(os.path.join(HERE, "data", "who", f"{name}.json")) as f:
        return json.load(f)


def show_one(models, tok, pair, kind):
    """Print the full answers for a single record, so the reader sees the
    model's actual words and not only a tally."""
    q, truth = pair["question"], value_of(pair["answer"])
    print(f"\n  {kind}")
    print(f"  Q: {q}")
    print(f"  true value: {truth}\n")
    for label, model in models.items():
        ans = ask_model(model, tok, q, max_new_tokens=40).strip()
        verdict = "STATES THE VALUE" if value_of(ans) == truth else "value not produced"
        print(f"    {label:<18} {ans[:70]}")
        print(f"    {'':<18} -> {verdict}")


def rate(model, tok, pairs, n):
    """Fraction of records whose true value the model still states. One
    question per distinct record, taken at a stride so we never sample the
    same entity twice."""
    hits = 0
    step = max(1, len(pairs) // n)
    picked = [pairs[i * step] for i in range(n)]
    for p in picked:
        ans = ask_model(model, tok, p["question"], max_new_tokens=40)
        hits += value_of(ans) == value_of(p["answer"])
    return hits, n


if __name__ == "__main__":
    ckpt = get_checkpoint_dir()
    need = ["trained_model.pt", "unlearned_npo.pt"]
    missing = [n for n in need if not os.path.exists(os.path.join(ckpt, n))]
    if missing:
        print(f"missing checkpoint(s): {missing}. Run ./run_pipeline.sh first.")
        sys.exit(1)

    print()
    rule()
    print("  FORGET-ME-NOT   behavioural proof of deletion".center(W))
    rule()
    print(f"\n  device: {DEVICE}")
    print("  loading the two models under comparison ...")
    tok = get_tokenizer()
    models = {
        "trained model": load_checkpoint(os.path.join(ckpt, "trained_model.pt")),
        "after unlearning": load_checkpoint(os.path.join(ckpt, "unlearned_npo.pt")),
    }
    for m in models.values():
        m.eval()
    print("    trained model      knows every record, including the deleted one")
    print("    after unlearning   the same model, one entity removed by NPO")

    forget, retain = load("forget"), load("retain")

    rule("-")
    print("  PART 1  one deleted record, in the model's own words")
    rule("-")
    show_one(models, tok, forget[0], "deleted record (was in the forget set)")

    rule("-")
    print("  PART 2  measured over several records, not one")
    rule("-")
    print("\n  Asking a single record proves little, so the same test is run")
    print("  across a sample of each set. These are the paper's two headline")
    print("  metrics, recomputed live.\n")

    N = 4
    after = models["after unlearning"]
    f_hit, f_n = rate(after, tok, forget, N)
    r_hit, r_n = rate(after, tok, retain, N)
    ues = 1 - f_hit / f_n
    krr = r_hit / r_n

    print(f"    deleted records still stating their value : {f_hit}/{f_n}")
    print(f"    retained records still answering correctly: {r_hit}/{r_n}")
    print()
    print(f"    UES  (forgetting, higher is better) = {ues:.2f}")
    print(f"    KRR  (retention,  higher is better) = {krr:.2f}")

    print()
    rule()
    print("  Forgetting is real, and it is not free: the same edit that")
    print("  removes the deleted values also costs some retained ones. The")
    print("  paper reports that cost instead of hiding it, at KRR 0.42 for")
    print("  this objective over three seeds. A loss curve shows none of it.")
    print()
    print("  Full per-record verdicts, the membership-inference attack and")
    print("  the relearning attack are in results/ .")
    rule()
    print()
