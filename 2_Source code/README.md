# Source Code

**Forget-Me-Not** — a generalised framework for trustworthy machine unlearning, and for proving whether the forgetting was real or fake.

The core question the code answers: **can a loss curve be trusted as proof that a model forgot something?** (Answer: no — which is why an audit and two adversarial attacks sit on top of the unlearning objectives themselves.)

## Layout

```
pipeline/          the framework itself — train, unlearn, audit, attack
web_console/        live dashboard: fetch, learn, ask, unlearn, verify
paper_figures/       scripts that render the manuscript's figures and tables
```

### `pipeline/`

| Stage | File | What it does |
|---|---|---|
| Data | `generate_dataset.py` | Builds the dataset — the one domain-specific adapter, swapped per domain |
| Model | `model.py` | GPT-style decoder-only transformer, from scratch, ~51M parameters |
| Data pipeline | `data.py` | Splits, tokenisation, answer-only loss masking |
| Train | `train.py` | Fine-tunes to verbatim memorisation |
| Unlearn | `unlearn_ga.py` | Gradient Ascent (baseline) + rollback guard |
| Unlearn | `unlearn_gd.py` | Gradient Difference — retain-anchored |
| Unlearn | `unlearn_npo.py` | Negative Preference Optimisation — bounded, self-limiting |
| Audit | `audit.py` | Five-probe behavioural audit, value-based leak detection |
| Reference | `oracle.py` / `compare.py` | Retrain-from-scratch gold standard + comparison |
| Attack | `attack_mia.py` | Membership inference (members vs. held-out) |
| Attack | `attack_relearn.py` | Relearning attack (hidden vs. erased) |
| Robustness | `sequential.py` / `sweep.py` | Sequential deletion + operating-point sweep |
| Deploy | `sync.py` / `sync_fetch.py` | Versioned-dataset learn/unlearn/retain cycle on live WHO data |
| Demo | `demo.py` | Terminal demonstration: learn, ask, delete, ask again |
| Demo | `verify_deletion.py` | 13-second behavioural proof of deletion — the script used in the execution videos |
| Output | `figures.py` | Figure generation for the exploratory (non-paper) result set |
| Run | `run_pipeline.sh` | One full pass: train → three unlearning methods → oracle → audit → attacks |
| Run | `run_multiseed.sh` | Repeats the pass for seeds 42, 43, 44 |

### Quick start

```bash
pip install -r pipeline/requirements.txt
cd pipeline
python generate_dataset.py         # build the dataset
EPOCHS=15 python train.py          # train to memorisation
bash run_pipeline.sh                # unlearn (3 methods) -> oracle -> audit -> attacks
python verify_deletion.py          # 13-second behavioural proof
```

Device (CUDA / Apple MPS / CPU) is auto-detected. All runs are seeded (`SEED=42`).

### `web_console/`

A live dashboard over the same pipeline: fetch a batch of real WHO records, learn them, ask the model anything, click a country or a single field to unlearn it, and watch the verification — before/after answers, live loss curves, erased/retained percentages, and the multi-seed benchmark. Every number shown is measured from the model's actual answers, never assumed.

```bash
python app.py
```

Then open `http://localhost:7861`. Requires a trained checkpoint (`sync_base.pt`), produced by `pipeline/sync.py`.

### `paper_figures/`

The scripts that generate the eleven figures in [`../1_Diagrams/`](../1_Diagrams/) and the multi-seed summary table quoted in the manuscript.

| File | Produces |
|---|---|
| `paper_figs.py` | The seven conceptual diagrams — Figures 1, 2, 3, 4, 5, 8, 11 |
| `aggregate_seeds.py` | The four data-driven figures — Figures 6, 7, 9, 10 — and `seeds_summary.txt` |
| `apply_template_format.py` | Converts the pandoc-generated `.docx` into the IEEE Access two-column layout |

## Applying it to a new domain

This is a **generalised framework**: the model, all three unlearning objectives, the reference model, the trustworthiness metrics and every attack are domain-independent. Moving to a new domain means changing exactly two things:

1. **`generate_dataset.py`** — emit `full.json` / `forget.json` / `retain.json` / `heldout.json` (lists of `{"question", "answer"}`) and `entities.json` (structured facts) for the new domain.
2. **`audit.py`** — adapt the probe templates and the leak check to the new domain's fields.

Everything downstream runs unchanged, since it operates on abstract forget/retain loaders. The only requirement is that the domain can be expressed as question–answer facts with structured entities. Section VII-F of the manuscript demonstrates exactly this: the identical pipeline, developed on a controlled benchmark, run unmodified on live WHO public-health statistics.

## Academic context

Consistent with TOFU (Maini et al., COLM 2024) and MUSE (Shi et al., ICLR 2025): approximate unlearning trades utility for forgetting rather than erasing cleanly, and loss-based evaluation cannot tell the difference. This project's contribution is a measurement layer — per-record verdicts, a composite trustworthiness score, a three-objective controlled comparison, and a relearning attack that separates hidden knowledge from erased knowledge — built and reproducible on a single consumer machine.

---
*Final-year AI/ML engineering capstone. PyTorch + a HuggingFace tokenizer only — the model, dataset pipeline, three unlearning objectives and audit are all built from scratch.*
