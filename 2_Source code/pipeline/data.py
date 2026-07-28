"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 2 — Dataset Loading (real WHO data)
=============================================================

The dataset is built from REAL WHO Global Health Observatory data by
generate_dataset.py: 100 countries, each with 5 real indicators (measles /
DTP3 / polio / BCG coverage + tuberculosis incidence), turned into 20
QA pairs per country = 2000 QA pairs.

Because the model is built FROM SCRATCH, it can only know a WHO figure
from our fine-tuning — so ground truth stays perfect and "did it forget?"
remains measurable, on live public-health data.

Splits (each country = one unit, like an author/patient in TOFU):
  - full    : all 2000 QA pairs (100 countries)   -> Phase 3 training
  - forget  : last 400 QA pairs  (20 countries)   -> Phase 4 unlearning
  - retain  : first 1600 QA pairs (80 countries)  -> utility control

Training format:  "Question: {q} Answer: {a}<|endoftext|>"

Loss masking: label = -100 on the question tokens and padding, so the
model is only trained (and later UN-trained) on the ANSWER tokens.

Run directly to sanity-check:  python data.py
"""

import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
MAX_LEN = 256          # longest WHO QA pair is well under 128 GPT-2 tokens
BATCH_SIZE = 16        # fits comfortably on a T4 with the 51M model
QA_PER_ENTITY = 20     # dataset structure: 20 consecutive questions per country

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "who")


# -------------------------------------------------------------
# Tokenizer (shared by all phases)
# -------------------------------------------------------------
def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token
    return tokenizer


# -------------------------------------------------------------
# Load WHO splits (local JSON, written by generate_dataset.py)
# -------------------------------------------------------------
def load_split(name):
    path = os.path.join(DATA_DIR, f"{name}.json")
    assert os.path.exists(path), \
        f"Missing {path} — run `python generate_dataset.py` first."
    with open(path) as f:
        return json.load(f)


def load_entities():
    """Structured per-country facts (measles, dtp3, polio, bcg, tb ...) — drives the audit."""
    return load_split("entities")


def load_who():
    full = load_split("full")
    forget = load_split("forget")
    retain = load_split("retain")

    print(f"[Phase 2]   full   : {len(full):>4} QA pairs "
          f"({len(full) // QA_PER_ENTITY} countries)")
    print(f"[Phase 2]   forget : {len(forget):>4} QA pairs "
          f"({len(forget) // QA_PER_ENTITY} countries)  <- to be unlearned")
    print(f"[Phase 2]   retain : {len(retain):>4} QA pairs "
          f"({len(retain) // QA_PER_ENTITY} countries)  <- must be preserved")

    assert len(forget) + len(retain) == len(full), "splits don't add up!"
    return full, forget, retain


# -------------------------------------------------------------
# Formatting
# -------------------------------------------------------------
def format_qa(question: str, answer: str) -> str:
    return f"Question: {question} Answer: {answer}"


# -------------------------------------------------------------
# PyTorch Dataset with answer-only loss masking
# -------------------------------------------------------------
class QADataset(Dataset):
    """
    Each item returns:
      input_ids : the full "Question: ... Answer: ...<eos>" token ids
      labels    : same ids, but -100 on the question part (and padding),
                  so cross-entropy only counts the answer tokens.
    """

    def __init__(self, rows, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        eos = tokenizer.eos_token_id
        for row in rows:
            prompt = f"Question: {row['question']} Answer:"
            full_text = format_qa(row["question"], row["answer"])

            prompt_ids = tokenizer(prompt).input_ids
            full_ids = tokenizer(full_text).input_ids + [eos]
            full_ids = full_ids[:max_len]

            labels = list(full_ids)
            n_prompt = min(len(prompt_ids), len(full_ids))
            labels[:n_prompt] = [-100] * n_prompt

            self.examples.append({
                "input_ids": full_ids,
                "labels": labels,
                "question": row["question"],
                "answer": row["answer"],
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch, pad_token_id):
    """Dynamic padding to the longest sequence in the batch (saves compute)."""
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels = [], []
    for ex in batch:
        n_pad = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * n_pad)
        labels.append(ex["labels"] + [-100] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# -------------------------------------------------------------
# DataLoader factory (used by Phases 3, 4, 6, 7, 8)
# -------------------------------------------------------------
def make_dataloaders(tokenizer, batch_size=BATCH_SIZE):
    print("[Phase 2] Loading WHO dataset (local, real GHO indicators)...")
    full, forget, retain = load_who()

    print("[Phase 2] Tokenizing (answer-only loss masking)...")
    full_ds = QADataset(full, tokenizer)
    forget_ds = QADataset(forget, tokenizer)
    retain_ds = QADataset(retain, tokenizer)

    collate_fn = lambda b: collate(b, tokenizer.pad_token_id)
    loaders = {
        "full":   DataLoader(full_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        "forget": DataLoader(forget_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        "retain": DataLoader(retain_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
    }
    print("[Phase 2] DataLoaders ready: "
          + ", ".join(f"{k}={len(v.dataset)} examples / {len(v)} batches" for k, v in loaders.items()))
    return loaders, {"full": full_ds, "forget": forget_ds, "retain": retain_ds}


# -------------------------------------------------------------
# Sanity test — run this file directly
# -------------------------------------------------------------
if __name__ == "__main__":
    tokenizer = get_tokenizer()
    loaders, datasets = make_dataloaders(tokenizer)

    ex = datasets["forget"].examples[0]
    print("\n[Phase 2] Sample from FORGET set:")
    print("  Q:", ex["question"])
    print("  A:", ex["answer"])

    batch = next(iter(loaders["forget"]))
    ids, labels = batch["input_ids"][0], batch["labels"][0]
    learned = ids[labels != -100]
    print("\n[Phase 2] Loss-masking check — tokens the model actually trains on:")
    print("  ", tokenizer.decode(learned)[:150])
    print("   (should be ONLY the answer text, no 'Question:' prefix)")

    print(f"\n[Phase 2] Batch shape: input_ids {tuple(batch['input_ids'].shape)}")
    print("\n[Phase 2] ✅ Data pipeline complete. Next: Phase 3 (training).")
