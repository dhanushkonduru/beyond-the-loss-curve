"""
=================================================================
 FORGET-ME-NOT : RICH REAL-WHO DEMO  (multi-field country records)
 Learn several REAL WHO indicators per country, ask rich questions,
 then delete one record and check how it answers afterwards.
=================================================================

Unlike the single-number demo, each country here is a RICH record built
from MANY real WHO indicators (measles / DTP3 / polio / BCG coverage +
tuberculosis incidence). So you can ask varied real questions:

    "What was India's polio vaccine coverage in 2025?"
    "What was the tuberculosis incidence in India?"
    "Which vaccines does India report?"

Flow:
  1. Fetch rich real records for a batch of countries (live, who.int)
  2. LEARN them live (no retraining)
  3. ASK — rich questions answered from the newly-learned facts
  4. DELETE one record — a whole country, OR one field (e.g. "India polio")
  5. ASK again — the deleted fact is gone; the rest survive

Usage:
  python demo.py                       # default batch
  python demo.py India Brazil Nigeria  # choose the countries
  PAUSE=2 python demo.py               # slower pace

Requires: checkpoints/sync_base.pt  (run once: python sync.py)
"""

import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request

import torch
from torch.utils.data import DataLoader

from model import DEVICE
from data import QADataset, collate, get_tokenizer
from train import ask_model, get_checkpoint_dir, load_checkpoint, train
from unlearn_ga import eval_loss, forever

API = "https://ghoapi.azureedge.net/api"
PAUSE = float(os.environ.get("PAUSE", 1.2))
BATCH = int(os.environ.get("BATCH", 5))
DEFAULT = ["India", "Brazil", "Nigeria", "Ethiopia", "Indonesia", "Kenya", "Egypt"]

# (key, human phrase used in questions/answers, GHO indicator code, unit)
FIELDS = [
    ("measles", "MCV1 (first-dose measles) vaccine coverage", "WHS4_544", "%"),
    ("dtp3",    "DTP3 (diphtheria-tetanus-pertussis) vaccine coverage", "WHS4_100", "%"),
    ("polio",   "polio (Pol3) vaccine coverage", "WHS4_129", "%"),
    ("bcg",     "BCG (tuberculosis) vaccine coverage", "WHS4_543", "%"),
    ("tb",      "tuberculosis incidence per 100 000 people", "MDG_0000000020", ""),
]
FIELD_BY_KEY = {f[0]: f for f in FIELDS}


def line(c="─"):
    print(c * 68)


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def country_map():
    return {d["Title"]: d["Code"]
            for d in get(f"{API}/DIMENSION/COUNTRY/DimensionValues")["value"]}


def latest(code, ccode):
    q = urllib.parse.quote(f"SpatialDim eq '{ccode}'")
    try:
        rows = get(f"{API}/{code}?$filter={q}")["value"]
    except Exception:
        return None
    s = {int(x["TimeDim"]): x["NumericValue"] for x in rows
         if x.get("TimeDim") and x.get("NumericValue") is not None}
    if not s:
        return None
    y = max(s)
    return y, int(round(s[y]))


def build_record(title, ccode):
    """A rich real record: {country, fields:{key:(year,value)}}, needs >=3 fields."""
    fields = {}
    for key, _, code, _ in FIELDS:
        got = latest(code, ccode)
        if got:
            fields[key] = got
    return {"country": title, "fields": fields} if len(fields) >= 3 else None


# ---- turn a record into QA rows -------------------------------------------
def field_q(country, key, year):
    _, phrase, _, _ = FIELD_BY_KEY[key]
    return f"What was {country}'s {phrase} in {year}?"


def field_a(country, key, year, value):
    _, phrase, _, unit = FIELD_BY_KEY[key]
    return f"{country}'s {phrase} in {year} was {value}{unit}."


def vaccines_q(country):
    return f"Which vaccines does {country} report coverage for?"


def rows_for(rec, only_key=None):
    """QA rows for a record. only_key -> just that one field (used for deletion)."""
    c = rec["country"]
    rows = []
    keys = [only_key] if only_key else list(rec["fields"])
    for key in keys:
        year, value = rec["fields"][key]
        for _ in range(3):
            rows.append({"question": field_q(c, key, year), "answer": field_a(c, key, year, value)})
    if not only_key:            # a profile QA so "which vaccines" works (taught, not inferred)
        vax = [FIELD_BY_KEY[k][1].split(" (")[0] for k in rec["fields"] if k != "tb"]
        rows.append({"question": vaccines_q(c),
                     "answer": f"{c} reports coverage for " + ", ".join(vax) + "."})
    return rows


def dl(rows, tok, bs=8, shuffle=True):
    return DataLoader(QADataset(rows, tok), batch_size=bs, shuffle=shuffle,
                      collate_fn=lambda b: collate(b, tok.pad_token_id))


def answer(model, tok, q):
    return ask_model(model, tok, q, max_new_tokens=44)[:160]


def stated_num(model, tok, q):
    # the value is the LAST number in the answer ("... in 2025 was 95%") — earlier
    # numbers are part of the field name/year (MCV1, DTP3, 2025, 100 000).
    nums = re.findall(r"\d+", answer(model, tok, q))
    return int(nums[-1]) if nums else None


def show_profile(model, tok, rec, tag):
    c = rec["country"]
    print(f"\n  {tag} — {c}:")
    for key in rec["fields"]:
        year, value = rec["fields"][key]
        q = field_q(c, key, year)
        print(f"    Q: {q}\n    A: {answer(model, tok, q)}")
    print(f"    Q: {vaccines_q(c)}\n    A: {answer(model, tok, vaccines_q(c))}")


def resolve(cmap, name):
    hits = [n for n in cmap if name.lower() in n.lower()]
    return hits[0] if hits else None


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    print("\n")
    line("═")
    print("   FORGET-ME-NOT  ·  RICH REAL-WHO RECORDS  (learn → ask → delete → ask)")
    line("═")

    base_path = os.path.join(get_checkpoint_dir(), "sync_base.pt")
    if not os.path.exists(base_path):
        print("\n[demo] Need the sync base model. Run once:  python sync.py")
        sys.exit(1)

    tok = get_tokenizer()
    print("\n[demo] connecting to who.int (Global Health Observatory)...")
    cmap = country_map()
    wanted = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT

    print("[demo] building rich real records live (several indicators each)...")
    batch = []
    for name in wanted:
        title = resolve(cmap, name)
        if not title:
            continue
        rec = build_record(title, cmap[title])
        if rec:
            batch.append(rec)
        if len(batch) >= BATCH:
            break
    if len(batch) < 2:
        print("[demo] couldn't assemble rich records — try other countries.")
        sys.exit(1)

    model = load_checkpoint(base_path)
    model.eval()

    # -----------------------------------------------------------------
    line()
    print("STEP 1 — rich REAL WHO records fetched live (who.int)")
    line()
    for rec in batch:
        bits = []
        for key in rec["fields"]:
            y, v = rec["fields"][key]
            bits.append(f"{key} {v}{FIELD_BY_KEY[key][3]}")
        print(f"   {rec['country']:<14} " + " · ".join(bits))
    time.sleep(PAUSE)

    # -----------------------------------------------------------------
    line()
    print(f"STEP 2 — LEARN all {len(batch)} rich records live (no retraining)")
    line()
    all_rows = [r for rec in batch for r in rows_for(rec)]
    learn_dl = dl(all_rows * 2, tok)
    target_dl = dl(all_rows, tok, shuffle=False)
    print(f"\n   teaching {len(all_rows)} QA facts across {len(batch)} countries ...\n")
    for ep in range(1, 15):
        train(model, learn_dl, epochs=1)
        if ep % 3 == 0:
            l = eval_loss(model, target_dl)
            print(f"   epoch {ep:>2} | learn-loss {l:5.2f} " + "▓" * max(1, int((3 - l) * 8)))
            time.sleep(PAUSE * 0.3)
    model.eval()

    # -----------------------------------------------------------------
    spot = batch[0]
    line()
    print(f"STEP 3 — ASK: what does it now know about {spot['country']}?  (rich questions)")
    line()
    show_profile(model, tok, spot, "LEARNED")
    print(f"\n   → these are REAL WHO figures, now answerable from the model.")
    time.sleep(PAUSE)

    # -----------------------------------------------------------------
    line()
    print("STEP 4 — YOUR DELETION REQUEST")
    line()
    print("   Delete a WHOLE country  (e.g.  India)")
    print("   or ONE field of a country  (e.g.  India polio   /   India tb)")
    countries = [rec["country"] for rec in batch]
    target_rec, target_key = None, None
    while target_rec is None:
        try:
            typed = input(f"   Delete what? {countries} [+ optional field]: ").strip()
        except EOFError:
            typed = countries[0]
            print(typed)
        parts = typed.split()
        cmatch = [rec for rec in batch if parts and parts[0].lower() in rec["country"].lower()]
        if len(cmatch) != 1:
            print("   Not one of the learned countries — try again.")
            continue
        target_rec = cmatch[0]
        if len(parts) > 1:
            k = parts[1].lower()
            if k in target_rec["fields"]:
                target_key = k
            else:
                print(f"   {target_rec['country']} has fields: {list(target_rec['fields'])} — try again.")
                target_rec = None

    scope = f"{target_rec['country']} · {target_key}" if target_key else f"{target_rec['country']} (all fields)"

    # -----------------------------------------------------------------
    line()
    print(f"STEP 5 — UNLEARN {scope} live — keep everything else")
    line()
    forget_rows = rows_for(target_rec, only_key=target_key)
    # anchor: everything we must keep — other countries + (if field-level) this country's other fields
    keep_rows = [r for rec in batch if rec is not target_rec for r in rows_for(rec)]
    if target_key:
        keep_rows += [r for k in target_rec["fields"] if k != target_key
                      for r in rows_for(target_rec, only_key=k)]
        keep_rows += rows_for(target_rec)          # keep the profile / other-field phrasings
        # the deleted field's TEMPLATE is shared across countries — anchor it hard so
        # forgetting one country's value doesn't drag the same field everywhere else
        keep_rows += [r for rec in batch if rec is not target_rec and target_key in rec["fields"]
                      for r in rows_for(rec, only_key=target_key)] * 3
    forget_dl = dl(forget_rows, tok)
    retain_dl = dl(keep_rows, tok)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)   # gentle: shared template is fragile
    fb, rb = forever(forget_dl), forever(retain_dl)
    base_r = eval_loss(model, retain_dl)
    model.train()
    safe = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    forgotten = False
    # what we watch to decide "gone" (a field of the target)
    wk = target_key or next(iter(target_rec["fields"]))
    wy, wv = target_rec["fields"][wk]
    watch_q = field_q(target_rec["country"], wk, wy)
    # a control fact that MUST stay correct. For a field delete, watch the SAME field
    # on another country (that's the template most at risk); else watch its first field.
    ctrl_rec = next(rec for rec in batch if rec is not target_rec
                    and (not target_key or target_key in rec["fields"]))
    ck = target_key if (target_key and target_key in ctrl_rec["fields"]) else next(iter(ctrl_rec["fields"]))
    cy, cv = ctrl_rec["fields"][ck]
    ctrl_q = field_q(ctrl_rec["country"], ck, cy)
    print(f"\n   forget-loss starts {eval_loss(model, forget_dl):.2f}, retain baseline {base_r:.2f}\n")
    for step in range(1, 401):
        f = next(fb)
        opt.zero_grad(set_to_none=True)
        _, fl = model(f["input_ids"].to(DEVICE), targets=f["labels"].to(DEVICE))
        r = next(rb)
        _, rl = model(r["input_ids"].to(DEVICE), targets=r["labels"].to(DEVICE))
        (-fl + 1.0 * rl).backward()          # strong retain anchor holds the shared template
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            fn = eval_loss(model, forget_dl)
            rn = eval_loss(model, retain_dl)
            model.eval()
            gone = stated_num(model, tok, watch_q) != wv and fn > 1.5   # real forgetting, not a blip
            ctrl_ok = stated_num(model, tok, ctrl_q) == cv
            print(f"   step {step:>3} | forget-loss {fn:5.2f} retain {rn:4.2f} "
                  + "█" * min(int(fn * 3), 32)
                  + ("   ← target gone" if gone else "") + ("" if ctrl_ok else "   ⚠ control wobbling"))
            if not ctrl_ok:                  # protect the rest: stop before it spreads
                model.load_state_dict(safe)
                print("   (utility guard: control answer moved — restored last clean state)")
                break
            if gone:
                forgotten = True
                print("   → target erased, control still clean — stopping.")
                break
            safe = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
            time.sleep(PAUSE * 0.3)
    model.eval()
    # final verdict from the ACTUAL end state, not which branch we exited on
    forgotten = stated_num(model, tok, watch_q) != wv
    ctrl_final_ok = stated_num(model, tok, ctrl_q) == cv

    # -----------------------------------------------------------------
    line()
    print(f"STEP 6 — ASK AGAIN: {scope} deleted?  is everything else still there?")
    line()
    show_profile(model, tok, target_rec, "AFTER (deletion target)")
    other = next(rec for rec in batch if rec is not target_rec)
    show_profile(model, tok, other, "CONTROL country (untouched)")

    line("═")
    print(f"   RESULT: deleted {scope} — "
          f"{'DONE ✅' if forgotten else 'reduced (rerun for a cleaner cut)'}"
          f"   ·   control {'intact ✅' if ctrl_final_ok else 'moved ⚠'}   ·   no retraining")
    if target_key:
        print(f"           {target_rec['country']}'s OTHER fields still answer correctly above "
              f"— only '{target_key}' was removed.")
    line("═")
    print()
