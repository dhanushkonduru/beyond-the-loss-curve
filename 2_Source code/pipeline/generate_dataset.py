"""
=============================================================
 FORGET-ME-NOT : WHO dataset builder  (generate_dataset.py)
=============================================================

Builds the project dataset from REAL WHO Global Health Observatory data.
Each country is one "record" with several real indicators; every indicator
becomes question-answer pairs the model is trained on, unlearned from, and
audited against — exactly the role the fictional patients used to play, but
now on live public-health data.

Because the model is built FROM SCRATCH, it can only know a WHO figure from
our training file — so ground truth is still perfect and "did it forget?"
stays measurable.

Indicators pulled live (one API call each, all countries at once):
  measles (MCV1), DTP3, polio (Pol3), BCG coverage, tuberculosis incidence.

Layout written to data/who/ (mirrors the old split names so the pipeline
is unchanged):
  full.json     all QA pairs                      -> train.py
  forget.json   FORGET countries' QA              -> unlearning
  retain.json   RETAIN countries' QA              -> utility control
  entities.json structured per-country facts      -> audit.py probes

Run:  python generate_dataset.py
"""

import json
import os
import urllib.parse
import urllib.request

API = "https://ghoapi.azureedge.net/api"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "who")

N_COUNTRIES = 100        # total countries in the dataset
N_FORGET = 20            # how many are the "forget" set (deletion requests)
PHRASINGS = 4            # QA phrasings per indicator -> QA_PER_ENTITY = 5*4 = 20

# (key, human phrase for Q/A, GHO indicator code, unit)
INDICATORS = [
    ("measles", "MCV1 (first-dose measles) vaccine coverage", "WHS4_544", "%"),
    ("dtp3",    "DTP3 (diphtheria-tetanus-pertussis) vaccine coverage", "WHS4_100", "%"),
    ("polio",   "polio (Pol3) vaccine coverage", "WHS4_129", "%"),
    ("bcg",     "BCG (tuberculosis) vaccine coverage", "WHS4_543", "%"),
    ("tb",      "tuberculosis incidence per 100 000 people", "MDG_0000000020", ""),
]

# question / answer phrasings (indexed 0..3); {ph}=indicator phrase
QT = [
    "What was {c}'s {ph} in {y}?",
    "State the {y} figure for {c}'s {ph}.",
    "In {y}, what was {c}'s {ph}?",
    "Report {c}'s {ph} for {y}.",
]
AT = [
    "{c}'s {ph} in {y} was {v}{u}.",
    "In {y}, {c}'s {ph} was {v}{u}.",
    "{c} recorded {v}{u} for {ph} in {y}.",
    "For {y}, {c}'s {ph} stood at {v}{u}.",
]


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def country_names():
    """Code -> Title, restricted to real countries (not WHO regions)."""
    return {d["Code"]: d["Title"]
            for d in get(f"{API}/DIMENSION/COUNTRY/DimensionValues")["value"]}


def fetch_indicator(code):
    """All countries' latest real value for one indicator -> {country_code: (year, value)}."""
    rows = get(f"{API}/{code}")["value"]
    series = {}
    for r in rows:
        cc, y, v = r.get("SpatialDim"), r.get("TimeDim"), r.get("NumericValue")
        if r.get("SpatialDimType") != "COUNTRY" or not cc or not y or not v:
            continue
        series.setdefault(cc, {})[int(y)] = int(round(v))
    return {cc: (max(s), s[max(s)]) for cc, s in series.items() if s}


def qa_rows(country, entity):
    """20 QA pairs for one country: 4 phrasings x 5 indicators."""
    rows = []
    for key, phrase, _, unit in INDICATORS:
        y, v = entity[key + "_year"], entity[key]
        for i in range(PHRASINGS):
            slot = {"c": country, "ph": phrase, "y": y, "v": v, "u": unit}
            rows.append({"question": QT[i].format(**slot),
                         "answer": AT[i].format(**slot)})
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("[WHO] fetching country list + 5 indicators live from who.int ...")
    names = country_names()
    series = {key: fetch_indicator(code) for key, _, code, _ in INDICATORS}
    for key, _, _, _ in INDICATORS:
        print(f"[WHO]   {key:<8} covers {len(series[key])} countries")

    # keep countries that have ALL five indicators and a real name
    keys = [k for k, *_ in INDICATORS]
    common = sorted(cc for cc in names
                    if all(cc in series[k] for k in keys))
    chosen = common[:N_COUNTRIES]
    print(f"[WHO] {len(common)} countries have all 5 indicators; using {len(chosen)}")

    entities, full = [], []
    for cc in chosen:
        country = names[cc]
        ent = {"country": country, "code": cc}
        for key in keys:
            y, v = series[key][cc]
            ent[key] = v
            ent[key + "_year"] = y
        entities.append(ent)
        full.extend(qa_rows(country, ent))

    # split: last N_FORGET countries are the deletion set (deterministic)
    forget_ents = entities[-N_FORGET:]
    retain_ents = entities[:-N_FORGET]
    forget_names = {e["country"] for e in forget_ents}
    forget = [r for e in forget_ents for r in qa_rows(e["country"], e)]
    retain = [r for e in retain_ents for r in qa_rows(e["country"], e)]

    # HELD-OUT countries (never trained on) -> true "non-members" for the MIA (phase 7)
    heldout_cc = common[N_COUNTRIES:N_COUNTRIES + N_FORGET]
    heldout = []
    for cc in heldout_cc:
        ent = {"country": names[cc], "code": cc}
        for key in keys:
            y, v = series[key][cc]
            ent[key], ent[key + "_year"] = v, y
        heldout.extend(qa_rows(names[cc], ent))

    def write(name, obj, overwrite=True):
        """Main splits are written idempotently (overwrite=False) so re-running to
        add the held-out set never disturbs data an existing model trained on."""
        path = os.path.join(OUT, f"{name}.json")
        if os.path.exists(path) and not overwrite:
            print(f"[WHO]   keeping existing {name}.json (unchanged)")
            return
        with open(path, "w") as f:
            json.dump(obj, f, indent=1)

    for name, obj in [("full", full), ("forget", forget),
                      ("retain", retain), ("entities", entities)]:
        write(name, obj, overwrite=False)
    write("heldout", heldout, overwrite=True)

    print(f"[WHO] data/who/  ->  full {len(full)} QA ({len(entities)} countries) | "
          f"forget {len(forget)} QA ({len(forget_ents)}) | retain {len(retain)} QA "
          f"({len(retain_ents)}) | heldout {len(heldout)} QA ({len(heldout_cc)})")
    print(f"[WHO] forget countries: {', '.join(sorted(forget_names))}")
    print("[WHO] ✅ done. Next:  EPOCHS=20 python train.py")
