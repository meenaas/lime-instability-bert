"""
LIME Instability Experiment - Round 3 (token-budgeted window + sample sweep)
=============================================================================
LegalPro-BERT on CUAD legal clauses. Measures how stable LIME's top-5 words
are across repeated runs (mean pairwise Jaccard@5), for Governing Law
(single-condition) versus Liquidated Damages (compositional) clauses.

WHAT CHANGED FROM ROUND 2, AND WHY
  1. Token-budgeted window. Round 2 padded the clause by 1,200 or 4,000
     CHARACTERS on each side, but the model only reads the first 384
     TOKENS. With 4,000 characters of left padding the gold clause starts
     at roughly token 800-1,000, i.e. outside the model's input. Round 3
     builds the window in token space so that the whole window, clause
     included, fits inside MAX_LEN. The script asserts this for every
     clause and records where the clause sits.
  2. Sample sweep. Round 2 used 150 LIME perturbation samples (LIME's
     default is 5,000). Round 3 repeats the stability measurement at
     several sample counts, so you can see whether instability is just
     sampling noise (Jaccard rises as samples increase) or persists.
  3. Confidence is re-checked on the NEW window. The >=0.5 filter is
     applied to the text LIME actually explains, not to an older window.
  4. More clauses. Instead of 5 hand-picked clauses, the script scans the
     held-out test split and takes up to N_PER_LABEL confident clauses
     per label, in a fixed (alphabetical) order.
  5. Lower-cased input. The model is uncased, but LIME splits on raw
     text, so "The"/"the" and "MEDICINES"/"Medicines" were separate LIME
     features in Round 2. Lower-casing removes those duplicates.
  6. Three extra diagnostics per run:
       - in_clause@5 : share of the top-5 words that occur in the gold
                       clause itself (a plausibility check)
       - R^2         : how well LIME's linear surrogate fits locally
       - chance      : the Jaccard@5 you would get from random top-5 lists
                       over the same vocabulary (a noise floor)
  7. One independent seed per run, passed to LIME directly.

REQUIRED FILES (same folder, or edit CONFIG / set the env variables)
  legalpro_bert_final_trainvaltest_best.pt   checkpoint: 'state_dict', 'labels'
  CUAD_v1.json                               CUAD v1
  split_record_TRAINVALTEST_PERMANENT.json   must contain 'test_context_hashes'

INSTALL
  pip install torch transformers lime scikit-learn numpy

HOW TO RUN
  # Full run. Use a GPU (e.g. Colab): about 45,000 model calls per clause.
  python3 lime_instability_round3.py

  # Quick smoke test first (2 clauses per label, small sweep, 2 runs each):
  N_PER_LABEL=2 SAMPLE_COUNTS=150,500 N_RUNS=2 python3 lime_instability_round3.py

  # CPU-friendly version (slower; drop the 5,000 setting):
  SAMPLE_COUNTS=150,500,1000,2500 N_PER_LABEL=5 python3 lime_instability_round3.py

  # Resumable: progress is saved after every LIME run. To work in batches:
  MAX_RUNS=20 python3 lime_instability_round3.py     # repeat until complete

OUTPUT FILES
  lime_round3_clauses.json    the selected clauses, windows, confidences
  lime_round3_progress.json   every run's top words and weights (resumable)
  lime_round3_summary.json    final table as JSON
  lime_round3_summary.csv     final table as CSV
"""

import os
import re
import csv
import json
import time
import hashlib
import itertools
import unicodedata

import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, AutoTokenizer
from transformers.utils import logging as hf_logging
from lime.lime_text import LimeTextExplainer

hf_logging.set_verbosity_error()

# ----------------------------------------------------------------------
# CONFIG (each value can also be set as an environment variable)
# ----------------------------------------------------------------------
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "legalpro_bert_final_trainvaltest_best.pt")
CUAD_PATH = os.environ.get("CUAD_PATH", "CUAD_v1.json")
SPLIT_RECORD_PATH = os.environ.get("SPLIT_RECORD_PATH", "split_record_TRAINVALTEST_PERMANENT.json")
TOKENIZER_NAME = os.environ.get("TOKENIZER", "bert-base-uncased")

CLAUSES_FILE = "lime_round3_clauses.json"
PROGRESS_FILE = "lime_round3_progress.json"
SUMMARY_JSON = "lime_round3_summary.json"
SUMMARY_CSV = "lime_round3_summary.csv"

LABELS_OF_INTEREST = ("Governing Law", "Liquidated Damages")
MAX_LEN = 384                                    # model input length, in tokens
N_RUNS = int(os.environ.get("N_RUNS", 5))        # LIME runs per clause per sample count
TOP_K = 5                                        # top-k words compared across runs
NUM_FEATURES = 10                                # words LIME reports per run
SAMPLE_COUNTS = [int(x) for x in os.environ.get("SAMPLE_COUNTS", "150,500,1000,2500,5000").split(",")]
N_PER_LABEL = int(os.environ.get("N_PER_LABEL", 10))   # max clauses per label
MIN_CONF = float(os.environ.get("MIN_CONF", 0.5))      # model P(gold label) on the window
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
BASE_SEED = int(os.environ.get("BASE_SEED", 3000))
MAX_RUNS_THIS_CALL = int(os.environ.get("MAX_RUNS", 10**9))
LOWERCASE = os.environ.get("LOWERCASE", "1") == "1"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DEVICE = os.environ.get("DEVICE", DEVICE)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------
# Helpers carried over unchanged from Round 2
# ----------------------------------------------------------------------
def normalize_context(s):
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()


def context_hash(ctx):
    return hashlib.sha1(normalize_context(ctx).encode("utf-8")).hexdigest()


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


class LegalProBERT(nn.Module):
    def __init__(self, config, num_labels=41, dropout=0.1):
        super().__init__()
        self.bert = BertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.pooler_output))


class ModelWrapper:
    """LIME-compatible predict_proba for one target label, batched."""

    def __init__(self, tokenizer, model, target_idx):
        self.tokenizer = tokenizer
        self.model = model
        self.target_idx = target_idx

    def predict_proba(self, texts):
        texts = [t if t.strip() else "[PAD]" for t in texts]   # LIME can produce empty strings
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            enc = self.tokenizer(texts[i:i + BATCH_SIZE], truncation=True, max_length=MAX_LEN,
                                 padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
                out.append(torch.sigmoid(logits)[:, self.target_idx].float().cpu().numpy())
        p = np.concatenate(out)
        return np.stack([1 - p, p], axis=1)


# ----------------------------------------------------------------------
# NEW: token-budgeted clause window
# ----------------------------------------------------------------------
def token_budget_window(tokenizer, ctx, char_start, char_end):
    """Return a window of `ctx` that contains the gold clause and fits in
    MAX_LEN tokens (including [CLS] and [SEP]), with the remaining budget
    split evenly between left and right context."""
    budget = MAX_LEN - 2
    enc = tokenizer(ctx, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = enc["offset_mapping"]
    n_tok = len(offsets)

    tok_s = next((i for i, (a, b) in enumerate(offsets) if b > char_start), n_tok - 1)
    tok_e = next((i for i, (a, b) in enumerate(offsets) if a >= char_end), n_tok)   # exclusive
    tok_e = max(tok_e, tok_s + 1)
    n_clause = tok_e - tok_s

    if n_clause >= budget:                      # clause alone is longer than the input
        lo, hi, truncated = tok_s, tok_s + budget, True
    else:
        spare = budget - n_clause
        left = min(spare // 2, tok_s)
        right = min(spare - left, n_tok - tok_e)
        left = min(spare - right, tok_s)        # give unused right budget back to the left
        lo, hi, truncated = tok_s - left, tok_e + right, False

    def cut(lo_, hi_):
        return ctx[offsets[lo_][0]:offsets[hi_ - 1][1]]

    window = cut(lo, hi)
    # Safety: re-tokenise the substring and trim the right edge if it is still too long.
    while len(tokenizer(window, add_special_tokens=True)["input_ids"]) > MAX_LEN and hi > tok_s + 1:
        hi -= 1
        window = cut(lo, hi)
        if hi < tok_e:
            truncated = True

    n_window = len(tokenizer(window, add_special_tokens=True)["input_ids"])
    assert n_window <= MAX_LEN, f"window is {n_window} tokens, exceeds MAX_LEN={MAX_LEN}"
    return {
        "window": window,
        "window_tokens": n_window,
        "clause_tokens": n_clause,
        "clause_first_token": tok_s - lo + 1,       # position inside the model input (after [CLS])
        "clause_last_token": min(tok_e, hi) - lo,
        "clause_truncated": truncated,
    }


def lime_words(text):
    """The vocabulary LIME sees (it splits on non-word characters)."""
    return {w for w in re.split(r"\W+", text) if w}


def chance_jaccard(vocab_size, k=TOP_K):
    """Expected Jaccard@k between two random size-k word sets from the vocabulary."""
    if vocab_size <= k:
        return 1.0
    overlap = k * k / vocab_size
    return overlap / (2 * k - overlap)


# ----------------------------------------------------------------------
def select_clauses(cuad, test_hashes, tokenizer, model, labels):
    """Scan the held-out test contracts and build a token-budgeted window for
    every Governing Law / Liquidated Damages clause. Keep those the model is
    confident about on that window."""
    candidates = []
    for contract in sorted(cuad, key=lambda c: c["title"]):
        para = contract["paragraphs"][0]
        ctx = para["context"]
        if context_hash(ctx) not in test_hashes:
            continue
        for qa in para["qas"]:
            label = qa["id"].split("__")[-1]
            if label not in LABELS_OF_INTEREST or qa.get("is_impossible") or not qa.get("answers"):
                continue
            ans = qa["answers"][0]
            cs, ce = ans["answer_start"], ans["answer_start"] + len(ans["text"])
            info = token_budget_window(tokenizer, ctx, cs, ce)
            text = info["window"].lower() if LOWERCASE else info["window"]
            clause_text = ans["text"].lower() if LOWERCASE else ans["text"]
            prob = float(ModelWrapper(tokenizer, model, labels.index(label)).predict_proba([text])[0, 1])
            vocab = lime_words(text)
            candidates.append({
                "key": f"{contract['title']}|{label}",
                "title": contract["title"],
                "label": label,
                "text": text,
                "clause_text": clause_text,
                "model_prob": round(prob, 4),
                "window_tokens": info["window_tokens"],
                "clause_tokens": info["clause_tokens"],
                "clause_first_token": info["clause_first_token"],
                "clause_last_token": info["clause_last_token"],
                "clause_truncated": info["clause_truncated"],
                "lime_vocab_size": len(vocab),
                "chance_jaccard": round(chance_jaccard(len(vocab)), 4),
            })

    selected = []
    for label in LABELS_OF_INTEREST:
        pool = [c for c in candidates if c["label"] == label]
        ok = [c for c in pool if c["model_prob"] >= MIN_CONF and not c["clause_truncated"]]
        log(f"{label}: {len(pool)} test clauses, {len(ok)} with P>={MIN_CONF} and fully inside the "
            f"window; using {min(len(ok), N_PER_LABEL)}")
        selected.extend(ok[:N_PER_LABEL])
    return candidates, selected


def summarise(selected, progress):
    rows = []
    for ns in SAMPLE_COUNTS:
        for label in LABELS_OF_INTEREST:
            per_clause_j, per_clause_in, per_clause_r2, chance = [], [], [], []
            for c in selected:
                if c["label"] != label:
                    continue
                runs = progress.get(c["key"], {}).get(str(ns), [])
                if len(runs) < 2:
                    continue
                tops = [r["top"] for r in runs]
                pairs = itertools.combinations(range(len(tops)), 2)
                per_clause_j.append(float(np.mean([jaccard(tops[i], tops[j]) for i, j in pairs])))
                clause_vocab = lime_words(c["clause_text"])
                per_clause_in.append(float(np.mean([np.mean([w in clause_vocab for w in t]) for t in tops])))
                r2 = [r["r2"] for r in runs if r.get("r2") is not None]
                if r2:
                    per_clause_r2.append(float(np.mean(r2)))
                chance.append(c["chance_jaccard"])
            if per_clause_j:
                rows.append({
                    "num_samples": ns,
                    "label": label,
                    "n_clauses": len(per_clause_j),
                    "mean_jaccard_at_5": round(float(np.mean(per_clause_j)), 4),
                    "std_across_clauses": round(float(np.std(per_clause_j)), 4),
                    "min": round(float(np.min(per_clause_j)), 4),
                    "max": round(float(np.max(per_clause_j)), 4),
                    "chance_level": round(float(np.mean(chance)), 4),
                    "in_clause_at_5": round(float(np.mean(per_clause_in)), 4),
                    "lime_r2": round(float(np.mean(per_clause_r2)), 4) if per_clause_r2 else None,
                })
    return rows


def main():
    for path in (CHECKPOINT_PATH, CUAD_PATH, SPLIT_RECORD_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    log(f"Device: {DEVICE} | sample counts: {SAMPLE_COUNTS} | runs per setting: {N_RUNS}")
    with open(SPLIT_RECORD_PATH) as f:
        test_hashes = set(json.load(f)["test_context_hashes"])
    with open(CUAD_PATH) as f:
        cuad = json.load(f)["data"]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    assert tokenizer.is_fast, "A fast tokenizer is required (for character offsets)."
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    labels = ckpt["labels"]
    config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072,
                        max_position_embeddings=512, type_vocab_size=2)
    if "config" in ckpt:                                   # optional override stored in the checkpoint
        config = BertConfig(**ckpt["config"])
    model = LegalProBERT(config, num_labels=len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    log("Model ready.")

    # ---- clause selection (cached so that resumed runs use the same clauses)
    if os.path.exists(CLAUSES_FILE):
        with open(CLAUSES_FILE) as f:
            saved = json.load(f)
        candidates, selected = saved["candidates"], saved["selected"]
        log(f"Loaded {len(selected)} previously selected clauses from {CLAUSES_FILE}.")
    else:
        candidates, selected = select_clauses(cuad, test_hashes, tokenizer, model, labels)
        with open(CLAUSES_FILE, "w") as f:
            json.dump({"max_len": MAX_LEN, "min_conf": MIN_CONF, "lowercase": LOWERCASE,
                       "candidates": candidates, "selected": selected}, f, indent=2)

    if not selected:
        log("No clauses passed the confidence filter. Lower MIN_CONF or check the checkpoint/tokenizer.")
        return

    print("\nSelected clauses (clause position is in tokens, inside the 384-token input):")
    for c in selected:
        print(f"  {c['label']:19s} | {c['title'][:42]:42s} | P={c['model_prob']:.2f} | "
              f"window={c['window_tokens']:3d} tok | clause at {c['clause_first_token']}-{c['clause_last_token']} "
              f"| LIME vocab={c['lime_vocab_size']}")
    print()

    # ---- LIME runs (resumable)
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)

    total_needed = len(selected) * len(SAMPLE_COUNTS) * N_RUNS
    done_this_call = 0
    stop = False
    for s_idx, ns in enumerate(SAMPLE_COUNTS):            # cheapest setting first
        for c_idx, c in enumerate(selected):
            runs = progress.setdefault(c["key"], {}).setdefault(str(ns), [])
            wrapper = ModelWrapper(tokenizer, model, labels.index(c["label"]))
            while len(runs) < N_RUNS:
                if done_this_call >= MAX_RUNS_THIS_CALL:
                    stop = True
                    break
                run_idx = len(runs)
                seed = BASE_SEED + 100000 * s_idx + 100 * c_idx + run_idx
                t0 = time.time()
                explainer = LimeTextExplainer(class_names=["other", "target"], random_state=seed)
                exp = explainer.explain_instance(c["text"], wrapper.predict_proba,
                                                 num_features=NUM_FEATURES, num_samples=ns, labels=(1,))
                pairs = exp.as_list(label=1)               # sorted by |weight|, largest first
                score = getattr(exp, "score", None)
                if isinstance(score, dict):
                    score = score.get(1)
                runs.append({
                    "seed": seed,
                    "top": [str(w) for w, _ in pairs[:TOP_K]],
                    "weights": [[str(w), round(float(v), 6)] for w, v in pairs],
                    "r2": None if score is None else round(float(score), 4),
                })
                done_this_call += 1
                with open(PROGRESS_FILE, "w") as f:
                    json.dump(progress, f, indent=1)
                log(f"n={ns:5d} | {c['key'][:50]:50s} | run {run_idx + 1}/{N_RUNS} "
                    f"({time.time() - t0:.0f}s): {runs[-1]['top']}")
            if stop:
                break
        if stop:
            break

    total_done = sum(min(len(progress.get(c["key"], {}).get(str(ns), [])), N_RUNS)
                     for c in selected for ns in SAMPLE_COUNTS)
    log(f"Progress: {total_done}/{total_needed} runs complete.")

    # ---- summary (printed for whatever is complete so far)
    rows = summarise(selected, progress)
    if rows:
        print("\n=== STABILITY BY SAMPLE COUNT ===")
        print(f"{'samples':>7} | {'label':19s} | {'n':>2} | {'Jaccard@5':>9} | {'std':>6} | "
              f"{'chance':>6} | {'in-clause@5':>11} | {'LIME R2':>7}")
        for r in rows:
            r2 = "   n/a" if r["lime_r2"] is None else f"{r['lime_r2']:7.3f}"
            print(f"{r['num_samples']:7d} | {r['label']:19s} | {r['n_clauses']:2d} | "
                  f"{r['mean_jaccard_at_5']:9.3f} | {r['std_across_clauses']:6.3f} | "
                  f"{r['chance_level']:6.3f} | {r['in_clause_at_5']:11.3f} | {r2}")
        with open(SUMMARY_JSON, "w") as f:
            json.dump({"config": {"max_len": MAX_LEN, "n_runs": N_RUNS, "top_k": TOP_K,
                                  "sample_counts": SAMPLE_COUNTS, "min_conf": MIN_CONF,
                                  "lowercase": LOWERCASE, "base_seed": BASE_SEED},
                       "rows": rows}, f, indent=2)
        with open(SUMMARY_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    if total_done >= total_needed:
        log("ALL RUNS COMPLETE.")
        print("""
HOW TO READ THE TABLE
  - Jaccard@5 close to 'chance' at every sample count  -> LIME's top words are
    essentially random for this model and input length, even when well sampled.
  - Jaccard@5 rising clearly with more samples         -> the Round 2 instability
    was mostly sampling noise. Report the curve; do not claim a structural mismatch
    from stability alone.
  - A gap between the two labels that persists at the highest sample count
    -> evidence for the original clause-type hypothesis.
  - in-clause@5 low even when Jaccard@5 is high        -> LIME is stable but points
    at words outside the gold clause, which is a separate (faithfulness) question.
""")
    else:
        log("Not finished yet - run the same command again to continue.")


if __name__ == "__main__":
    main()
