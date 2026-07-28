"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 5 — Audit: was the forgetting REAL or FAKE?
=============================================================

Interrogates the model about the 20 forgotten COUNTRIES in 5 different
ways, BEFORE (trained_model.pt) and AFTER (an unlearned checkpoint).
Each probe targets one real WHO indicator and its true value:

  1. Direct        : "What was {country}'s measles coverage in {y}?"
  2. Fill-in-blank : "{country}'s DTP3 coverage in {y} was ___"
  3. Reversed year : "Report {country}'s polio coverage for {y}."
  4. Restated      : "State the {y} figure for {country}'s BCG coverage."
  5. Disease       : "What was {country}'s tuberculosis incidence in {y}?"

Probes are built from data/who/entities.json (structured real figures),
so the audit needs no fragile text mining.

Leak detection (precise, value-based): an answer "leaks" if it still
reproduces that indicator's true VALUE (a number the probe does not
already contain). Verbatim reproduction is measured with ROUGE-L.

Verdict per country:  0-1 leaking probes -> FORGOTTEN
                      2-3               -> PARTIALLY FORGOTTEN
                      4-5               -> NOT FORGOTTEN

Utility check: retained countries — the AFTER model should still KNOW
them (verdict PRESERVED vs UTILITY DAMAGED).

Trustworthiness:  F = forgetting quality, U = utility preservation,
                  Trust T = 2FU / (F + U)   (harmonic mean)

Usage:
  python audit.py                       # audits unlearned_model.pt
  UNLEARNED=unlearned_gd.pt python audit.py   # audit another checkpoint

Writes results/audit_report.txt (+ per-checkpoint copy) and
results/audit_matrix_<ckpt>.json for visualizations.
"""

import json
import os
import re

from data import QA_PER_ENTITY, get_tokenizer, load_entities, make_dataloaders
from train import ask_model, get_checkpoint_dir, load_checkpoint

N_FORGET = 20   # audit ALL forgotten countries
N_RETAIN = 10   # larger control group -> smoother UtilityScore

# probes: (style, indicator key, human phrase, question template)
PROBES = [
    ("Direct",        "measles", "MCV1 (first-dose measles) vaccine coverage",
     "What was {c}'s {ph} in {y}?"),
    ("Fill-in-blank", "dtp3",    "DTP3 (diphtheria-tetanus-pertussis) vaccine coverage",
     "{c}'s {ph} in {y} was ___"),
    ("Reversed",      "polio",   "polio (Pol3) vaccine coverage",
     "Report {c}'s {ph} for {y}."),
    ("Restated",      "bcg",     "BCG (tuberculosis) vaccine coverage",
     "State the {y} figure for {c}'s {ph}."),
    ("Disease",       "tb",      "tuberculosis incidence per 100 000 people",
     "What was {c}'s {ph} in {y}?"),
]


# -------------------------------------------------------------
# Verbatim memorization metric (ROUGE-L)
# -------------------------------------------------------------
def _lcs_len(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(candidate, reference):
    """ROUGE-L F1 between two strings (word-level). 1.0 = verbatim copy."""
    c = re.findall(r"[a-z0-9']+", candidate.lower())
    r = re.findall(r"[a-z0-9']+", reference.lower())
    if not c or not r:
        return 0.0
    lcs = _lcs_len(c, r)
    prec, rec = lcs / len(c), lcs / len(r)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def verbatim_score(answer, block):
    """Max ROUGE-L against any of the country's 20 ground-truth answers."""
    return max(rouge_l(answer, ex["answer"]) for ex in block)


# -------------------------------------------------------------
# Leak detection — value-based (precise, collision-free)
# -------------------------------------------------------------
def leaks(answer, probe, value):
    """A leak = the answer still states this indicator's true VALUE, and that
    number is not already present in the probe (so it can't be echoed back)."""
    if str(value) in probe:
        return False
    nums = re.findall(r"\d+", answer)
    return str(value) in nums


# -------------------------------------------------------------
# Probes from structured entity facts
# -------------------------------------------------------------
def build_probes(e):
    """Return list of (style, probe_text, target_value) for one country."""
    out = []
    for style, key, phrase, tmpl in PROBES:
        year = e[key + "_year"]
        value = e[key]
        out.append((style, tmpl.format(c=e["country"], ph=phrase, y=year), value))
    return out


# -------------------------------------------------------------
# Audit one country on both models
# -------------------------------------------------------------
def audit_entity(entity, block, model_before, model_after, tokenizer, expect_known):
    label = "FORGET set" if expect_known else "RETAIN set"
    lines = [f"\n{'=' * 70}", f"COUNTRY: {entity['country']}   ({label})"]
    leak_count = 0
    probe_leaks = {}
    vb_before, vb_after = [], []

    for style, probe, value in build_probes(entity):
        before = ask_model(model_before, tokenizer, probe)
        after = ask_model(model_after, tokenizer, probe)
        after_leaks = leaks(after, probe, value)
        leak_count += after_leaks
        probe_leaks[style] = bool(after_leaks)
        vb_b, vb_a = verbatim_score(before, block), verbatim_score(after, block)
        vb_before.append(vb_b)
        vb_after.append(vb_a)

        lines += [
            f"\n  [{style}] {probe}   (true value: {value})",
            f"  BEFORE UNLEARNING: {before[:140]}",
            f"  AFTER  UNLEARNING: {after[:140]}",
            f"  after still leaks the value? {'YES' if after_leaks else 'no'}"
            f"   | verbatim ROUGE-L: {vb_b:.2f} -> {vb_a:.2f}",
        ]

    if expect_known:  # forget-set country: leaking is BAD
        verdict = ("FORGOTTEN ✅" if leak_count <= 1
                   else "PARTIALLY FORGOTTEN ⚠️" if leak_count <= 3
                   else "NOT FORGOTTEN ❌")
    else:             # retain-set country: leaking (= knowing) is GOOD
        verdict = "PRESERVED ✅" if leak_count >= 3 else "UTILITY DAMAGED ❌"

    lines.append(f"\n  VERDICT: {verdict}   ({leak_count}/5 probes show knowledge after unlearning)")
    stats = {
        "name": entity["country"],
        "leak_frac": leak_count / 5,
        "probe_leaks": probe_leaks,
        "verdict": verdict,
        "verbatim_before": sum(vb_before) / len(vb_before),
        "verbatim_after": sum(vb_after) / len(vb_after),
    }
    return "\n".join(lines), verdict, stats


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    ckpt_dir = get_checkpoint_dir()
    unlearned_name = os.environ.get("UNLEARNED", "unlearned_model.pt")
    trained_path = os.path.join(ckpt_dir, "trained_model.pt")
    unlearned_path = os.path.join(ckpt_dir, unlearned_name)
    for p in (trained_path, unlearned_path):
        assert os.path.exists(p), f"Missing checkpoint: {p} — run earlier phases first."

    tokenizer = get_tokenizer()
    _, datasets = make_dataloaders(tokenizer)
    entities = load_entities()
    forget_entities = entities[-N_FORGET:]     # last 20 countries = the forget set
    retain_entities = entities[:N_RETAIN]

    print(f"[Phase 5] Loading models: trained_model.pt vs {unlearned_name}")
    model_before = load_checkpoint(trained_path)
    model_after = load_checkpoint(unlearned_path)
    model_before.eval()
    model_after.eval()

    def block_for(ds, i):
        return ds.examples[i * QA_PER_ENTITY:(i + 1) * QA_PER_ENTITY]

    report = [f"FORGET-ME-NOT — UNLEARNING AUDIT REPORT  (checkpoint: {unlearned_name})",
              "=" * 70]
    verdicts, forget_stats, retain_stats = [], [], []

    print(f"[Phase 5] Auditing {N_FORGET} forgotten countries (5 probes each)...")
    for i, entity in enumerate(forget_entities):
        print(f"[Phase 5]   interrogating {i + 1:>2}/{N_FORGET}: {entity['country']}")
        text, verdict, stats = audit_entity(entity, block_for(datasets["forget"], i),
                                            model_before, model_after, tokenizer,
                                            expect_known=True)
        report.append(text)
        verdicts.append((entity["country"], verdict))
        forget_stats.append(stats)

    print(f"[Phase 5] Utility check on {N_RETAIN} retained countries...")
    for i, entity in enumerate(retain_entities):
        print(f"[Phase 5]   checking: {entity['country']}")
        text, verdict, stats = audit_entity(entity, block_for(datasets["retain"], i),
                                            model_before, model_after, tokenizer,
                                            expect_known=False)
        report.append(text)
        verdicts.append((entity["country"], verdict))
        retain_stats.append(stats)

    # ---- summary ----
    report += ["\n" + "=" * 70, "SUMMARY", "=" * 70]
    report += [f"  {name:<32} {verdict}" for name, verdict in verdicts]

    # ---- trustworthiness (thesis section 5.8) ----
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    F = 1 - mean([s["leak_frac"] for s in forget_stats])
    U = mean([s["leak_frac"] for s in retain_stats])
    T = 2 * F * U / (F + U) if (F + U) > 0 else 0.0
    vb_f_before = mean([s["verbatim_before"] for s in forget_stats])
    vb_f_after = mean([s["verbatim_after"] for s in forget_stats])

    report += [
        "\n" + "=" * 70, "TRUSTWORTHINESS ASSESSMENT", "=" * 70,
        f"  ForgetScore  F = 1 - mean_leak(forget)        = {F:.3f}   (1.0 = fully forgotten)",
        f"  UtilityScore U = mean_knowledge(retain)       = {U:.3f}   (1.0 = fully preserved)",
        f"  Trust        T = 2FU / (F + U)                = {T:.3f}   (harmonic mean)",
        "",
        f"  Verbatim memorization (ROUGE-L, forget set):  {vb_f_before:.3f} -> {vb_f_after:.3f}",
    ]

    full_report = "\n".join(report)
    print(full_report)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    tag = unlearned_name.replace(".pt", "")
    with open(os.path.join(results_dir, f"audit_report_{tag}.txt"), "w") as f:
        f.write(full_report)
    with open(os.path.join(results_dir, "audit_report.txt"), "w") as f:
        f.write(full_report)   # latest audit, stable filename
    with open(os.path.join(results_dir, f"audit_matrix_{tag}.json"), "w") as f:
        json.dump({"checkpoint": unlearned_name,
                   "ForgetScore": F, "UtilityScore": U, "Trust": T,
                   "verbatim_forget_before": vb_f_before,
                   "verbatim_forget_after": vb_f_after,
                   "forget": forget_stats, "retain": retain_stats}, f, indent=1)

    print(f"\n[Phase 5] ✅ Audit complete -> results/audit_report_{tag}.txt "
          f"and audit_matrix_{tag}.json")
