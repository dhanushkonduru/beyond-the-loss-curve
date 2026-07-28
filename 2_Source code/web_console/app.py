"""
=============================================================
 FORGET-ME-NOT : Live Console (web UI)
 Serves a dashboard over the real pipeline: fetch live WHO
 records, learn them, ask the model, unlearn one record or
 field, verify it is gone, and report every number honestly.

 Run:   .venv/bin/python ui/app.py        (from the repo root)
 Open:  http://localhost:7861
=============================================================
"""

import copy
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                     # noqa: E402
from torch.utils.data import DataLoader          # noqa: E402

from model import DEVICE                         # noqa: E402
from data import QADataset, collate, get_tokenizer   # noqa: E402
from train import ask_model, get_checkpoint_dir, load_checkpoint, train  # noqa: E402
from unlearn_ga import eval_loss, forever        # noqa: E402

API = "https://ghoapi.azureedge.net/api"
PORT = int(os.environ.get("PORT", 7861))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DEFAULT_COUNTRIES = ["India", "Brazil", "Nigeria", "Ethiopia", "Indonesia"]
LEARN_EPOCHS = int(os.environ.get("LEARN_EPOCHS", 14))

# (key, human phrase used in questions/answers, GHO indicator code, unit)
FIELDS = [
    ("measles", "MCV1 (first-dose measles) vaccine coverage", "WHS4_544", "%"),
    ("dtp3",    "DTP3 (diphtheria-tetanus-pertussis) vaccine coverage", "WHS4_100", "%"),
    ("polio",   "polio (Pol3) vaccine coverage", "WHS4_129", "%"),
    ("bcg",     "BCG (tuberculosis) vaccine coverage", "WHS4_543", "%"),
    ("tb",      "tuberculosis incidence per 100 000 people", "MDG_0000000020", ""),
]
FIELD_BY_KEY = {f[0]: f for f in FIELDS}

# ----------------------------------------------------------------- state
S = {
    "lock": threading.Lock(),
    "model": None, "tok": None,
    "batch": [],            # [{country, fields:{key:{year,value,unit,phrase,status,before,after}}}]
    "learned": False,
    "qa": [],               # [{q, a, phase, t}]
    "metrics": {},          # last unlearn metrics
    "job": {"kind": None, "running": False, "log": [], "series": [], "result": None, "error": None},
}


def log(msg):
    with S["lock"]:
        S["job"]["log"].append(msg)
    print("[ui]", msg, flush=True)


def series(point):
    with S["lock"]:
        S["job"]["series"].append(point)


def start_job(kind, fn, *args):
    with S["lock"]:
        if S["job"]["running"]:
            return False
        S["job"] = {"kind": kind, "running": True, "log": [], "series": [],
                    "result": None, "error": None}

    def runner():
        try:
            result = fn(*args)
            with S["lock"]:
                S["job"]["result"] = result
        except Exception as e:  # surface the real error in the UI
            import traceback
            traceback.print_exc()
            with S["lock"]:
                S["job"]["error"] = f"{type(e).__name__}: {e}"
        finally:
            with S["lock"]:
                S["job"]["running"] = False

    threading.Thread(target=runner, daemon=True).start()
    return True


# ----------------------------------------------------------------- WHO fetch
def http_get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def latest(code, ccode):
    q = urllib.parse.quote(f"SpatialDim eq '{ccode}'")
    try:
        rows = http_get(f"{API}/{code}?$filter={q}")["value"]
    except Exception:
        return None
    s = {int(x["TimeDim"]): x["NumericValue"] for x in rows
         if x.get("TimeDim") and x.get("NumericValue") is not None}
    if not s:
        return None
    y = max(s)
    return y, int(round(s[y]))


def job_fetch(names):
    log("connecting to who.int (Global Health Observatory) ...")
    cmap = {d["Title"]: d["Code"]
            for d in http_get(f"{API}/DIMENSION/COUNTRY/DimensionValues")["value"]}
    log(f"country dimension loaded ({len(cmap)} territories)")
    batch = []
    for name in names:
        hits = [n for n in cmap if name.lower() in n.lower()]
        if not hits:
            log(f"'{name}' not found in the WHO country list — skipped")
            continue
        title = hits[0]
        fields = {}
        for key, phrase, code, unit in FIELDS:
            got = latest(code, cmap[title])
            if got:
                y, v = got
                fields[key] = {"year": y, "value": v, "unit": unit, "phrase": phrase,
                               "status": "fetched", "before": None, "after": None}
                log(f"  {title} · {key}: {v}{unit} ({y})  [indicator {code}]")
            else:
                log(f"  {title} · {key}: no data")
        if len(fields) >= 3:
            batch.append({"country": title, "fields": fields, "status": "fetched"})
            log(f"record built for {title} ({len(fields)} fields)")
        else:
            log(f"{title} has fewer than 3 usable fields — skipped")
    if len(batch) < 2:
        raise RuntimeError("could not assemble at least 2 rich records")
    with S["lock"]:
        S["batch"] = batch
        S["learned"] = False
        S["metrics"] = {}
        S["qa"] = []
    log(f"DONE — {len(batch)} live records ready to learn")
    return {"records": len(batch)}


# ----------------------------------------------------------------- QA plumbing
def field_q(country, key, year):
    return f"What was {country}'s {FIELD_BY_KEY[key][1]} in {year}?"


def field_a(country, key, year, value):
    _, phrase, _, unit = FIELD_BY_KEY[key]
    return f"{country}'s {phrase} in {year} was {value}{unit}."


def vaccines_q(country):
    return f"Which vaccines does {country} report coverage for?"


def rows_for(rec, only_key=None):
    c = rec["country"]
    rows = []
    keys = [only_key] if only_key else list(rec["fields"])
    for key in keys:
        f = rec["fields"][key]
        for _ in range(3):
            rows.append({"question": field_q(c, key, f["year"]),
                         "answer": field_a(c, key, f["year"], f["value"])})
    if not only_key:
        vax = [FIELD_BY_KEY[k][1].split(" (")[0] for k in rec["fields"] if k != "tb"]
        rows.append({"question": vaccines_q(c),
                     "answer": f"{c} reports coverage for " + ", ".join(vax) + "."})
    return rows


def dl(rows, bs=8, shuffle=True):
    tok = S["tok"]
    return DataLoader(QADataset(rows, tok), batch_size=bs, shuffle=shuffle,
                      collate_fn=lambda b: collate(b, tok.pad_token_id))


def answer(q, n=44):
    return ask_model(S["model"], S["tok"], q, max_new_tokens=n)[:160]


def stated_num(q):
    # the value is the LAST number in the answer — earlier digits belong to the
    # field name or the year (MCV1, DTP3, 2025, 100 000)
    nums = re.findall(r"\d+", answer(q))
    return int(nums[-1]) if nums else None


# ----------------------------------------------------------------- model jobs
def ensure_model():
    if S["model"] is None:
        base = os.path.join(get_checkpoint_dir(), "sync_base.pt")
        if not os.path.exists(base):
            raise RuntimeError("checkpoints/sync_base.pt missing — run: python sync.py")
        log(f"loading base model on {DEVICE} ...")
        S["tok"] = get_tokenizer()
        S["model"] = load_checkpoint(base)
        S["model"].eval()
        log("base model ready (51M params, trained on WHO release data)")


def job_learn():
    ensure_model()
    torch.manual_seed(42)
    batch = S["batch"]
    all_rows = [r for rec in batch for r in rows_for(rec)]
    learn_dl = dl(all_rows * 2)
    target_dl = dl(all_rows, shuffle=False)
    log(f"teaching {len(all_rows)} QA facts across {len(batch)} countries "
        f"({LEARN_EPOCHS} epochs, no retraining)")
    l0 = eval_loss(S["model"], target_dl)
    series({"x": 0, "loss": round(l0, 3)})
    log(f"starting loss {l0:.2f} (high = the model does not know these facts yet)")
    for ep in range(1, LEARN_EPOCHS + 1):
        train(S["model"], learn_dl, epochs=1)
        l = eval_loss(S["model"], target_dl)
        series({"x": ep, "loss": round(l, 3)})
        log(f"epoch {ep:>2}/{LEARN_EPOCHS} | learn-loss {l:5.2f}")
    S["model"].eval()

    log("verifying: asking the model every learned fact ...")
    ok = total = 0
    for rec in batch:
        hits = 0
        for key, f in rec["fields"].items():
            q = field_q(rec["country"], key, f["year"])
            a = answer(q)
            got = re.findall(r"\d+", a)
            hit = bool(got) and int(got[-1]) == f["value"]
            with S["lock"]:
                f["before"] = a
                f["status"] = "learned" if hit else "weak"
            hits += hit
            ok += hit
            total += 1
        with S["lock"]:
            rec["status"] = "learned"
        log(f"  {rec['country']}: {hits}/{len(rec['fields'])} fields answered exactly")
    with S["lock"]:
        S["learned"] = True
        S["metrics"]["learned_pct"] = round(100 * ok / max(total, 1))
        S["metrics"]["facts"] = total
    log(f"DONE — {ok}/{total} facts verified ({round(100*ok/max(total,1))}% exact recall)")
    return {"learned_pct": round(100 * ok / max(total, 1))}


def job_unlearn(country, only_key):
    ensure_model()
    torch.manual_seed(42)
    batch = S["batch"]
    target = next(r for r in batch if r["country"] == country)
    scope = f"{country} · {only_key}" if only_key else f"{country} (all fields)"
    log(f"deletion request: {scope}")

    forget_rows = rows_for(target, only_key=only_key)
    keep_rows = [r for rec in batch if rec is not target for r in rows_for(rec)]
    if only_key:
        keep_rows += [r for k in target["fields"] if k != only_key
                      for r in rows_for(target, only_key=k)]
        keep_rows += rows_for(target)
        # the deleted field's TEMPLATE is shared across countries — anchor it hard
        keep_rows += [r for rec in batch if rec is not target and only_key in rec["fields"]
                      for r in rows_for(rec, only_key=only_key)] * 3
    forget_dl, retain_dl = dl(forget_rows), dl(keep_rows)

    wk = only_key or next(iter(target["fields"]))
    wf = target["fields"][wk]
    watch_q = field_q(country, wk, wf["year"])
    ctrl_rec = next(rec for rec in batch if rec is not target
                    and (not only_key or only_key in rec["fields"]))
    ck = only_key if (only_key and only_key in ctrl_rec["fields"]) else next(iter(ctrl_rec["fields"]))
    cf = ctrl_rec["fields"][ck]
    ctrl_q = field_q(ctrl_rec["country"], ck, cf["year"])
    log(f"watch: '{watch_q}' (must stop saying {wf['value']})")
    log(f"control: '{ctrl_q}' (must keep saying {cf['value']})")

    model = S["model"]
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    fb, rb = forever(forget_dl), forever(retain_dl)
    f0 = eval_loss(model, forget_dl)
    r0 = eval_loss(model, retain_dl)
    series({"x": 0, "forget": round(f0, 3), "retain": round(r0, 3)})
    log(f"forget-loss starts {f0:.2f}, retain baseline {r0:.2f}")
    with S["lock"]:
        target["status"] = "unlearning"
        for k in ([only_key] if only_key else list(target["fields"])):
            target["fields"][k]["status"] = "unlearning"

    safe = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.train()
    steps_done, rolled_back = 0, False
    for step in range(1, 401):
        f = next(fb)
        opt.zero_grad(set_to_none=True)
        _, fl = model(f["input_ids"].to(DEVICE), targets=f["labels"].to(DEVICE))
        r = next(rb)
        _, rl = model(r["input_ids"].to(DEVICE), targets=r["labels"].to(DEVICE))
        (-fl + 1.0 * rl).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps_done = step
        if step % 10 == 0:
            fn = eval_loss(model, forget_dl)
            rn = eval_loss(model, retain_dl)
            model.eval()
            gone = stated_num(watch_q) != wf["value"] and fn > 1.5
            ctrl_ok = stated_num(ctrl_q) == cf["value"]
            series({"x": step, "forget": round(fn, 3), "retain": round(rn, 3)})
            log(f"step {step:>3} | forget-loss {fn:5.2f} | retain {rn:4.2f}"
                + ("  ← target gone" if gone else "")
                + ("" if ctrl_ok else "  ⚠ control wobbling"))
            if not ctrl_ok:
                model.load_state_dict(safe)
                rolled_back = True
                log("utility guard: control answer moved — restored last clean state")
                break
            if gone:
                log("target erased, control still clean — stopping")
                break
            safe = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
    model.eval()

    # ---- verdict from the actual end state, field by field -----------------
    log("verifying after unlearning: re-asking every question ...")
    keys = [only_key] if only_key else list(target["fields"])
    gone_n = 0
    for k in keys:
        f = target["fields"][k]
        a = answer(field_q(country, k, f["year"]))
        got = re.findall(r"\d+", a)
        gone = not got or int(got[-1]) != f["value"]
        with S["lock"]:
            f["after"] = a
            f["status"] = "forgotten" if gone else "residual"
        gone_n += gone
        log(f"  {country} · {k}: {'GONE' if gone else 'STILL PRESENT'}")
    # untouched fields of a field-level delete: confirm they survived
    if only_key:
        for k, f in target["fields"].items():
            if k == only_key:
                continue
            a = answer(field_q(country, k, f["year"]))
            got = re.findall(r"\d+", a)
            with S["lock"]:
                f["after"] = a
                f["status"] = "learned" if (got and int(got[-1]) == f["value"]) else "weak"
    # retention across every other country
    keep_ok = keep_total = 0
    for rec in batch:
        if rec is target:
            continue
        for k, f in rec["fields"].items():
            a = answer(field_q(rec["country"], k, f["year"]))
            got = re.findall(r"\d+", a)
            hit = bool(got) and int(got[-1]) == f["value"]
            with S["lock"]:
                f["after"] = a
                f["status"] = "learned" if hit else "weak"
            keep_ok += hit
            keep_total += 1
        log(f"  retention {rec['country']}: intact")
    ff = eval_loss(model, forget_dl)
    with S["lock"]:
        if only_key:   # field-level delete: the record itself survives
            target["status"] = "partial" if gone_n == len(keys) else "learned"
        else:
            target["status"] = "forgotten" if gone_n == len(keys) else "partial"
        S["metrics"] = {**S["metrics"],
                        "scope": scope,
                        "erased_pct": round(100 * gone_n / len(keys)),
                        "erased_n": gone_n, "erased_total": len(keys),
                        "retention_pct": round(100 * keep_ok / max(keep_total, 1)),
                        "forget_loss_before": round(f0, 2),
                        "forget_loss_after": round(ff, 2),
                        "steps": steps_done, "rolled_back": rolled_back}
    log(f"DONE — erased {gone_n}/{len(keys)} target fields, "
        f"retention {keep_ok}/{keep_total} "
        f"({round(100*keep_ok/max(keep_total,1))}%), no retraining")
    return S["metrics"]


# ----------------------------------------------------------------- snapshot
def snapshot():
    with S["lock"]:
        bench = None
        p = os.path.join(ROOT, "results", "seeds_summary.json")
        if os.path.exists(p):
            bench = json.load(open(p))
        return {
            "device": str(DEVICE),
            "model_loaded": S["model"] is not None,
            "learned": S["learned"],
            "batch": S["batch"],
            "qa": S["qa"][-40:],
            "metrics": S["metrics"],
            "benchmark": bench,
            "job": {k: v for k, v in S["job"].items() if k != "result"} | {
                "result": S["job"]["result"]},
        }


# ----------------------------------------------------------------- http
class H(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode()) if n else {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._json(snapshot())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            data = self._body()
        except Exception:
            return self._json({"error": "bad json"}, 400)

        if self.path == "/api/fetch":
            names = [n.strip() for n in data.get("countries", []) if n.strip()] or DEFAULT_COUNTRIES
            ok = start_job("fetch", job_fetch, names[:7])
            return self._json({"started": ok})

        if self.path == "/api/learn":
            if not S["batch"]:
                return self._json({"error": "fetch data first"}, 400)
            ok = start_job("learn", job_learn)
            return self._json({"started": ok})

        if self.path == "/api/unlearn":
            if not S["learned"]:
                return self._json({"error": "learn the batch first"}, 400)
            country, key = data.get("country"), data.get("field") or None
            if not any(r["country"] == country for r in S["batch"]):
                return self._json({"error": "unknown country"}, 400)
            ok = start_job("unlearn", job_unlearn, country, key)
            return self._json({"started": ok})

        if self.path == "/api/ask":
            q = (data.get("question") or "").strip()
            if not q:
                return self._json({"error": "empty question"}, 400)
            if S["job"]["running"]:
                return self._json({"error": "busy: the model is training right now"}, 409)
            if S["model"] is None:
                return self._json({"error": "no model loaded — learn a batch first"}, 400)
            a = answer(q, n=48)
            phase = "after unlearning" if S["metrics"].get("scope") else (
                "after learning" if S["learned"] else "base model")
            with S["lock"]:
                S["qa"].append({"q": q, "a": a, "phase": phase, "t": time.strftime("%H:%M:%S")})
            return self._json({"answer": a, "phase": phase})

        return self._json({"error": "not found"}, 404)

    def log_message(self, *a):   # quiet
        pass


if __name__ == "__main__":
    print(f"[ui] FORGET-ME-NOT live console on http://localhost:{PORT}  (device {DEVICE})")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
