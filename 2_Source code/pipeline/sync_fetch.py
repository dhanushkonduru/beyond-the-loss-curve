"""
=============================================================
 FORGET-ME-NOT : Dataset-Model Sync Framework
 sync_fetch.py — build two dataset "releases" from REAL WHO data
=============================================================

Pulls real WHO Global Health Observatory data — Measles-containing
vaccine, 1st dose (MCV1) immunization coverage among 1-year-olds,
indicator WHS4_544 — which is public, aggregated, country-level and
openly licensed. Values are REAL WHO numbers.

It then builds two versioned releases to drive the sync framework:

  release_2023 : country coverage for years 2018-2022
  release_2024 : adds the real 2023 figures (records to LEARN),
                 and drops a set of countries' records
                 (records to UNLEARN — simulating estimates WHO
                 withdrew), everything else KEPT.

HONEST NOTE: the coverage *values* are genuine WHO data. The
release-to-release churn (which records are added vs. dropped) is
constructed here to exercise the framework, because the GHO API
exposes only the current revision, not historical published snapshots.

Output -> data/whosync/ : release_2023.json, release_2024.json,
          records_2023.json, records_2024.json, diff.json

Run (needs internet):  python sync_fetch.py
"""

import json
import os
import random
import urllib.parse
import urllib.request

API = "https://ghoapi.azureedge.net/api"
INDICATOR = "WHS4_544"   # MCV1 coverage among 1-year-olds (%)
YEARS_OLD = list(range(2018, 2023))   # 2018-2022 -> release_2023
NEW_YEAR = 2023                        # -> added in release_2024
N_DROP_COUNTRIES = 15                  # records withdrawn in release_2024
rng = random.Random(42)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "whosync")


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def country_names():
    data = get(f"{API}/DIMENSION/COUNTRY/DimensionValues")
    return {d["Code"]: d["Title"] for d in data["value"]}


def fetch_records():
    print("[WHO] fetching real MCV1 coverage (WHS4_544)...")
    names = country_names()
    q = urllib.parse.quote("SpatialDimType eq 'COUNTRY'")
    rows = get(f"{API}/{INDICATOR}?$filter={q}")["value"]
    recs = {}
    for r in rows:
        code, year, val = r["SpatialDim"], r.get("TimeDim"), r.get("NumericValue")
        if code in names and year and val is not None:
            recs[(code, year)] = {"country": names[code], "code": code,
                                  "year": int(year), "value": int(round(val))}
    print(f"[WHO]   {len(recs)} country-year records, "
          f"{len({k[0] for k in recs})} countries")
    return recs


# ---- QA rendering (varied phrasing, same style as MedForget) ----
QT = ["What was the MCV1 immunization coverage in {c} in {y}?",
      "What percentage of 1-year-olds in {c} received the first measles dose in {y}?",
      "State the {y} measles (MCV1) coverage for {c}.",
      "In {y}, what was {c}'s first-dose measles vaccine coverage?"]
AT = ["In {y}, MCV1 immunization coverage in {c} was {v}%.",
      "{c} reported {v}% first-dose measles coverage in {y}.",
      "Coverage of MCV1 in {c} for {y} stood at {v}%.",
      "In {y}, {v}% of 1-year-olds in {c} received their first measles dose."]


def to_qa(rec, k):
    r = random.Random(k)
    v = {"c": rec["country"], "y": rec["year"], "v": rec["value"]}
    return {"question": r.choice(QT).format(**v), "answer": r.choice(AT).format(**v),
            "country": rec["country"], "year": rec["year"], "value": rec["value"],
            "id": f"{rec['code']}-{rec['year']}"}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    recs = fetch_records()

    countries = sorted({k[0] for k in recs})
    dropped = set(rng.sample(countries, min(N_DROP_COUNTRIES, len(countries))))

    rel_2023 = [to_qa(v, i) for i, (k, v) in enumerate(sorted(recs.items()))
                if v["year"] in YEARS_OLD]
    rel_2024 = [to_qa(v, i) for i, (k, v) in enumerate(sorted(recs.items()))
                if (v["year"] in YEARS_OLD or v["year"] == NEW_YEAR)
                and v["code"] not in dropped]

    ids_2023 = {r["id"] for r in rel_2023}
    ids_2024 = {r["id"] for r in rel_2024}
    diff = {
        "added":   sorted(ids_2024 - ids_2023),   # LEARN (mostly the new 2023 year)
        "removed": sorted(ids_2023 - ids_2024),   # UNLEARN (withdrawn countries)
        "kept":    sorted(ids_2023 & ids_2024),   # RETAIN
        "dropped_countries": sorted(dropped),
    }

    for fname, data in [("release_2023.json", rel_2023), ("release_2024.json", rel_2024),
                        ("records_2023.json", [{k: r[k] for k in ('id','country','year','value')}
                                               for r in rel_2023]),
                        ("records_2024.json", [{k: r[k] for k in ('id','country','year','value')}
                                               for r in rel_2024]),
                        ("diff.json", diff)]:
        with open(os.path.join(OUT, fname), "w") as f:
            json.dump(data, f, indent=1)

    print(f"\n[WHO] release_2023: {len(rel_2023)} records")
    print(f"[WHO] release_2024: {len(rel_2024)} records")
    print(f"[WHO] diff -> LEARN {len(diff['added'])}, "
          f"UNLEARN {len(diff['removed'])}, RETAIN {len(diff['kept'])}")
    print("[WHO] sample:", rel_2023[0]["question"], "->", rel_2023[0]["answer"])
    print("[WHO] ✅ real WHO releases written ->", OUT)
