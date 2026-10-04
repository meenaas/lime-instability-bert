"""
Round 4 - What is the classifier relying on?  (diagnostics first, then LIME)
=============================================================================
Uses the SAME inputs as the paper's Table 1: for each of the 77 held-out test
contracts, the first 512 tokens of
    context (full text = lead), extractive_summary, pegasus_summary, hybrid_leakfree.

STAGE 1 (always runs, a few minutes)
  1. Reproduces Table 1 on all 77 contracts (micro/macro P, R, F1 per system).
  2. Adds two baselines that never read the contract: CONSTANT predicts every
     label whose training-set frequency is above a cut-off chosen on validation;
     PREDICT EVERY LABEL predicts all 41 labels for every contract.
  3. Per-label table for one system (default: context):
       n_pos        test contracts with the label
       visible      share of those where the gold clause text is inside the
                    512-token input the model actually reads
       F1_model / F1_const
       AUROC        does the model's probability separate contracts with the
                    label from those without?  0.5 = no better than chance
       p_pos/p_neg  mean probability for contracts with / without the label

STAGE 2 (only with RUN_LIME=1)
  LIME stability sweep on labels where the experiment is meaningful: the model
  separates positives from negatives (AUROC) AND the evidence is in its input.
  Labels are chosen automatically from the Stage 1 table, or set LIME_LABELS.
  REQUIRE_VISIBLE=0 also explains labels whose gold clause is OUTSIDE the input
  (e.g. Governing Law), to see which words the model uses instead. Two columns
  describe where LIME's top-5 words come from:
       in_clause@5  share found in the gold clause text
       in_head@5    share found in the first 50 words of the input (title/parties)

REQUIRED FILES (same folder)
  legalpro_bert_final_trainvaltest_best.pt
  CUADv1.json   (or CUAD_v1.json - either name is found)
  split_record_TRAINVALTEST_PERMANENT.json
  hybrid_leakfree_TEST_57.csv
  missing20_hybrid_leakfree.csv

RUN
  pip install torch transformers lime scikit-learn pandas numpy
  python3 round4_diagnose_and_lime.py                       # Stage 1 only
  RUN_LIME=1 python3 round4_diagnose_and_lime.py            # Stage 1 + LIME
  RUN_LIME=1 LIME_LABELS="Agreement Date,License Grant" python3 round4_diagnose_and_lime.py
  # quick LIME smoke test:
  RUN_LIME=1 N_PER_LABEL=1 SAMPLE_COUNTS=150,500 N_RUNS=2 python3 round4_diagnose_and_lime.py

OUTPUT
  round4_table1.csv, round4_per_label.csv, round4_probs.npz
  lime_round4_clauses.json, lime_round4_progress.json, lime_round4_summary.csv
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
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import BertConfig, BertModel, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# ----------------------------------------------------------------------
# CONFIG (environment variables override)
# ----------------------------------------------------------------------
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "legalpro_bert_final_trainvaltest_best.pt")
SPLIT_RECORD_PATH = os.environ.get("SPLIT_RECORD_PATH", "split_record_TRAINVALTEST_PERMANENT.json")
TEST57_CSV = os.environ.get("TEST57_CSV", "hybrid_leakfree_TEST_57.csv")
MISSING20_CSV = os.environ.get("MISSING20_CSV", "missing20_hybrid_leakfree.csv")
CUAD_PATH = os.environ.get("CUAD_PATH") or next(
    (p for p in ("CUADv1.json", "CUAD_v1.json") if os.path.exists(p)), "CUADv1.json")
# Training used the tokenizer of this model; bert-base-uncased is the fallback (same 30,522 vocab).
TOKENIZERS = [os.environ["TOKENIZER"]] if os.environ.get("TOKENIZER") else \
    ["AmitTewari/LegalPro-BERT-base", "bert-base-uncased"]

SYSTEMS = ["context", "extractive_summary", "pegasus_summary", "hybrid_leakfree"]
PRIMARY = os.environ.get("PRIMARY_SYSTEM", "context")     # system used for per-label table and LIME
MAX_LEN = 512
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))

RUN_LIME = os.environ.get("RUN_LIME", "0") == "1"
LIME_LABELS = [s.strip() for s in os.environ.get("LIME_LABELS", "").split(",") if s.strip()]
MAX_AUTO_LABELS = int(os.environ.get("MAX_AUTO_LABELS", 3))
MIN_AUROC = float(os.environ.get("MIN_AUROC", 0.70))
MIN_VISIBLE = float(os.environ.get("MIN_VISIBLE", 0.50))
REQUIRE_VISIBLE = os.environ.get("REQUIRE_VISIBLE", "1") == "1"   # 0 = also explain labels whose clause is outside the input
HEAD_WORDS = int(os.environ.get("HEAD_WORDS", 50))                 # "opening of the contract" for the in_head metric
N_PER_LABEL = int(os.environ.get("N_PER_LABEL", 6))
N_RUNS = int(os.environ.get("N_RUNS", 5))
SAMPLE_COUNTS = [int(x) for x in os.environ.get("SAMPLE_COUNTS", "150,500,1000,2500,5000").split(",")]
TOP_K, NUM_FEATURES = 5, 10
BASE_SEED = int(os.environ.get("BASE_SEED", 4000))
MAX_RUNS_THIS_CALL = int(os.environ.get("MAX_RUNS", 10**9))

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
# Same helpers as the training notebook / reproduce_full77_table1.py
# ----------------------------------------------------------------------
def normalize_context(s):
    if not isinstance(s, str):
        s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def context_hash(ctx):
    return hashlib.sha1(normalize_context(ctx).encode("utf-8")).hexdigest()


def label_from_id(qa_id):
    return qa_id.split("__")[-1].strip() if "__" in qa_id else qa_id.strip()


def load_cuad(path, clause_list):
    """{hash: 41-dim label vector} and {hash: {label: [gold span texts]}}, de-duplicated."""
    j = json.load(open(path))
    data = j["data"] if "data" in j else j
    idx = {c: i for i, c in enumerate(clause_list)}
    labels, spans = {}, {}
    for contract in data:
        for p in contract.get("paragraphs", []):
            h = context_hash(p["context"])
            if h in labels:
                continue
            y = np.zeros(len(clause_list), dtype=int)
            sp = {}
            for qa in p["qas"]:
                c = label_from_id(qa["id"])
                if not qa.get("is_impossible", False) and c in idx:
                    y[idx[c]] = 1
                    sp.setdefault(c, []).extend(a["text"] for a in qa.get("answers", []))
            labels[h], spans[h] = y, sp
    return labels, spans


class LegalProBERT(nn.Module):
    def __init__(self, config, num_labels=41, dropout=0.1):
        super().__init__()
        self.bert = BertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.pooler_output))


@torch.no_grad()
def predict_all(model, tokenizer, texts):
    """Probabilities for all labels; texts are truncated to MAX_LEN tokens, as in the paper."""
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [str(t) if str(t).strip() else "[PAD]" for t in texts[i:i + BATCH_SIZE]]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def visible_part(tokenizer, text):
    """The part of `text` the model actually reads: its first MAX_LEN-2 tokens."""
    text = str(text)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    off = enc["offset_mapping"]
    if len(off) <= MAX_LEN - 2:
        return text
    return text[:off[MAX_LEN - 3][1]]


def lime_words(text):
    return {w for w in re.split(r"\W+", text) if w}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def chance_jaccard(v, k=TOP_K):
    if v <= k:
        return 1.0
    o = k * k / v
    return o / (2 * k - o)


def prf(y_true, y_pred):
    return {
        "MiP": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "MiR": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "MiF1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "MaF1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ----------------------------------------------------------------------
def main():
    for p in (CHECKPOINT_PATH, SPLIT_RECORD_PATH, TEST57_CSV, MISSING20_CSV, CUAD_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")
    log(f"Device: {DEVICE}")

    # ---- model + tokenizer
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    clause_list, thr = ckpt["labels"], float(ckpt["best_thr"])
    config = BertConfig(**ckpt["config"]) if "config" in ckpt else BertConfig(
        vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        intermediate_size=3072, max_position_embeddings=512, type_vocab_size=2)
    model = LegalProBERT(config, num_labels=len(clause_list))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    tokenizer = None
    for name in TOKENIZERS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
            log(f"Tokenizer: {name} (vocab {tokenizer.vocab_size})")
            break
        except Exception as e:                                      # noqa: BLE001
            log(f"Could not load tokenizer {name}: {type(e).__name__}")
    assert tokenizer is not None and tokenizer.is_fast, "No fast tokenizer could be loaded."
    log(f"Checkpoint: threshold={thr:.2f}, stored test micro-F1={ckpt.get('test_micro_f1')}, "
        f"macro-F1={ckpt.get('test_macro_f1')}")

    # ---- data
    split = json.load(open(SPLIT_RECORD_PATH))
    gold, spans = load_cuad(CUAD_PATH, clause_list)
    df = pd.concat([pd.read_csv(TEST57_CSV)[["key_hash"] + SYSTEMS],
                    pd.read_csv(MISSING20_CSV)[["key_hash"] + SYSTEMS]], ignore_index=True)
    assert set(df["key_hash"]) == set(split["test_context_hashes"]), "CSV contracts != test split"
    assert all(context_hash(c) == h for h, c in zip(df["key_hash"], df["context"])), "hash mismatch"
    y_test = np.stack([gold[h] for h in df["key_hash"]])
    y_train = np.stack([gold[h] for h in split["train_context_hashes"]])
    y_val = np.stack([gold[h] for h in split["val_context_hashes"]])
    log(f"Contracts: train={len(y_train)} val={len(y_val)} test={len(y_test)}; labels={len(clause_list)}")

    # ================= STAGE 1 =================
    # constant baseline: predict label if its TRAIN frequency >= cut-off (cut-off chosen on VAL)
    prev = y_train.mean(axis=0)
    cut = max(np.arange(0.05, 0.96, 0.05),
              key=lambda t: f1_score(y_val, np.tile(prev >= t, (len(y_val), 1)), average="micro"))
    const_pred = np.tile(prev >= cut, (len(y_test), 1)).astype(int)

    probs, rows = {}, []
    for s in SYSTEMS:
        probs[s] = predict_all(model, tokenizer, df[s].fillna("").tolist())
        rows.append({"system": s, **prf(y_test, (probs[s] >= thr).astype(int)),
                     "avg_labels_predicted": float((probs[s] >= thr).sum(1).mean())})
    rows.append({"system": f"CONSTANT (no input; train freq >= {cut:.2f})", **prf(y_test, const_pred),
                 "avg_labels_predicted": float(const_pred.sum(1).mean())})
    all_pred = np.ones_like(y_test)
    rows.append({"system": "PREDICT EVERY LABEL (no input)", **prf(y_test, all_pred),
                 "avg_labels_predicted": float(all_pred.sum(1).mean())})
    np.savez("round4_probs.npz", y_test=y_test, labels=np.array(clause_list),
             hashes=df["key_hash"].to_numpy(), **{f"probs_{s}": probs[s] for s in SYSTEMS})

    print(f"\n=== TABLE 1 on all {len(y_test)} test contracts (threshold {thr:.2f}) ===")
    print(f"{'system':46s} {'MiP':>6} {'MiR':>6} {'MiF1':>6} {'MaF1':>6} {'labels/contract':>16}")
    for r in rows:
        print(f"{r['system']:46s} {r['MiP']:6.3f} {r['MiR']:6.3f} {r['MiF1']:6.3f} {r['MaF1']:6.3f} "
              f"{r['avg_labels_predicted']:16.1f}")
    print(f"(true labels per contract: {y_test.sum(1).mean():.1f})")
    pd.DataFrame(rows).round(4).to_csv("round4_table1.csv", index=False)

    # per-label table for the primary system
    vis_text = [normalize_context(visible_part(tokenizer, t)) for t in df[PRIMARY].fillna("")]
    P = probs[PRIMARY]
    pred = (P >= thr).astype(int)
    per = []
    for j, lab in enumerate(clause_list):
        pos = y_test[:, j] == 1
        vis = [any(normalize_context(sp)[:80] in vis_text[i] for sp in spans[df["key_hash"][i]].get(lab, []))
               for i in np.where(pos)[0]]
        per.append({
            "label": lab,
            "n_pos": int(pos.sum()),
            "train_freq": round(float(prev[j]), 3),
            "visible": round(float(np.mean(vis)), 3) if len(vis) else np.nan,
            "F1_model": round(f1_score(y_test[:, j], pred[:, j], zero_division=0), 3),
            "F1_const": round(f1_score(y_test[:, j], const_pred[:, j], zero_division=0), 3),
            "AUROC": round(roc_auc_score(y_test[:, j], P[:, j]), 3) if 0 < pos.sum() < len(pos) else np.nan,
            "p_pos": round(float(P[pos, j].mean()), 3) if pos.any() else np.nan,
            "p_neg": round(float(P[~pos, j].mean()), 3) if (~pos).any() else np.nan,
        })
    per_df = pd.DataFrame(per).sort_values("n_pos", ascending=False)
    per_df.to_csv("round4_per_label.csv", index=False)
    print(f"\n=== PER-LABEL, system = {PRIMARY} ===")
    print(per_df.to_string(index=False))
    print("""
HOW TO READ STAGE 1
  - If the CONSTANT row matches or beats a system on MiF1, micro-F1 is being carried by
    label frequency, not by reading the text. PREDICT EVERY LABEL is the matching floor
    for MaF1. What the model adds beyond both shows up in the per-label AUROC.
  - 'visible' near 0 with a high F1_model means the label is predicted without its clause
    being in the input (e.g. a clause that sits at the end of contracts).
  - AUROC near 0.5 means the model's score for that label does not depend on the contract.
""")

    if not RUN_LIME:
        log("Stage 1 complete. Re-run with RUN_LIME=1 for the LIME sweep.")
        return

    # ================= STAGE 2: LIME =================
    from lime.lime_text import LimeTextExplainer

    if LIME_LABELS:
        chosen = [l for l in LIME_LABELS if l in clause_list]
        missing = [l for l in LIME_LABELS if l not in clause_list]
        if missing:
            log(f"Unknown labels ignored: {missing}")
    else:
        ok = per_df[(per_df["AUROC"] >= MIN_AUROC) & (per_df["visible"] >= MIN_VISIBLE) & (per_df["n_pos"] >= 5)]
        chosen = ok.sort_values("AUROC", ascending=False)["label"].head(MAX_AUTO_LABELS).tolist()
    if not chosen:
        log(f"No label has AUROC >= {MIN_AUROC} and visible >= {MIN_VISIBLE}. Nothing meaningful for LIME "
            f"to explain on '{PRIMARY}'. Try PRIMARY_SYSTEM=extractive_summary, or set LIME_LABELS.")
        return
    log(f"LIME labels: {chosen}")

    CLAUSES, PROGRESS = "lime_round4_clauses.json", "lime_round4_progress.json"
    if os.path.exists(CLAUSES):
        selected = json.load(open(CLAUSES))["selected"]
        log(f"Loaded {len(selected)} previously selected items from {CLAUSES}.")
    else:
        selected = []
        for lab in chosen:
            j = clause_list.index(lab)
            cands = []
            for i in range(len(df)):
                h = df["key_hash"][i]
                gold_sp = [normalize_context(s) for s in spans[h].get(lab, [])]
                if y_test[i, j] != 1 or P[i, j] < thr:
                    continue                                     # true positives only
                is_visible = any(s[:80] in vis_text[i] for s in gold_sp)
                if REQUIRE_VISIBLE and not is_visible:
                    continue                                     # evidence must be in the input
                text = visible_part(tokenizer, df[PRIMARY][i]).lower()
                cands.append({"key": f"{h[:10]}|{lab}", "hash": h, "label": lab, "text": text,
                              "clause_text": " ".join(gold_sp), "clause_visible": bool(is_visible),
                              "head_text": " ".join(text.split()[:HEAD_WORDS]), "model_prob": round(float(P[i, j]), 4),
                              "lime_vocab_size": len(lime_words(text)),
                              "chance_jaccard": round(chance_jaccard(len(lime_words(text))), 4)})
            cands.sort(key=lambda c: c["hash"])
            log(f"{lab}: {len(cands)} true positives{' with visible evidence' if REQUIRE_VISIBLE else ''}; using {min(len(cands), N_PER_LABEL)}")
            selected.extend(cands[:N_PER_LABEL])
        json.dump({"system": PRIMARY, "threshold": thr, "selected": selected}, open(CLAUSES, "w"), indent=1)
    if not selected:
        log("No items selected.")
        return

    class Wrapper:
        def __init__(self, j):
            self.j = j

        def predict_proba(self, texts):
            p = predict_all(model, tokenizer, list(texts))[:, self.j]
            return np.stack([1 - p, p], axis=1)

    progress = json.load(open(PROGRESS)) if os.path.exists(PROGRESS) else {}
    total = len(selected) * len(SAMPLE_COUNTS) * N_RUNS
    done_now, stop = 0, False
    for s_idx, ns in enumerate(SAMPLE_COUNTS):
        for c_idx, c in enumerate(selected):
            runs = progress.setdefault(c["key"], {}).setdefault(str(ns), [])
            w = Wrapper(clause_list.index(c["label"]))
            while len(runs) < N_RUNS:
                if done_now >= MAX_RUNS_THIS_CALL:
                    stop = True
                    break
                seed = BASE_SEED + 100000 * s_idx + 100 * c_idx + len(runs)
                t0 = time.time()
                exp = LimeTextExplainer(class_names=["other", "target"], random_state=seed).explain_instance(
                    c["text"], w.predict_proba, num_features=NUM_FEATURES, num_samples=ns, labels=(1,))
                pairs = exp.as_list(label=1)
                score = getattr(exp, "score", None)
                score = score.get(1) if isinstance(score, dict) else score
                runs.append({"seed": seed, "top": [str(x) for x, _ in pairs[:TOP_K]],
                             "weights": [[str(x), round(float(v), 6)] for x, v in pairs],
                             "r2": None if score is None else round(float(score), 4)})
                done_now += 1
                json.dump(progress, open(PROGRESS, "w"), indent=1)
                log(f"n={ns:5d} | {c['key']:40s} | run {len(runs)}/{N_RUNS} ({time.time() - t0:.0f}s): {runs[-1]['top']}")
            if stop:
                break
        if stop:
            break

    out = []
    for ns in SAMPLE_COUNTS:
        for lab in chosen:
            J, IN, HD, R2, CH = [], [], [], [], []
            for c in selected:
                runs = progress.get(c["key"], {}).get(str(ns), [])
                if c["label"] != lab or len(runs) < 2:
                    continue
                tops = [r["top"] for r in runs]
                J.append(np.mean([jaccard(tops[a], tops[b]) for a, b in itertools.combinations(range(len(tops)), 2)]))
                cv = lime_words(c["clause_text"])
                IN.append(np.mean([np.mean([x in cv for x in t]) for t in tops]))
                hv = lime_words(c.get("head_text", ""))
                HD.append(np.mean([np.mean([x in hv for x in t]) for t in tops]))
                R2.extend(r["r2"] for r in runs if r.get("r2") is not None)
                CH.append(c["chance_jaccard"])
            if J:
                out.append({"num_samples": ns, "label": lab, "n_items": len(J),
                            "jaccard_at_5": round(float(np.mean(J)), 4), "std": round(float(np.std(J)), 4),
                            "chance": round(float(np.mean(CH)), 4), "in_clause_at_5": round(float(np.mean(IN)), 4),
                            "in_head_at_5": round(float(np.mean(HD)), 4),
                            "lime_r2": round(float(np.mean(R2)), 4) if R2 else None})
    if out:
        print(f"\n=== LIME STABILITY BY SAMPLE COUNT (system = {PRIMARY}) ===")
        print(pd.DataFrame(out).to_string(index=False))
        with open("lime_round4_summary.csv", "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            wri.writeheader()
            wri.writerows(out)
    n_done = sum(min(len(progress.get(c["key"], {}).get(str(ns), [])), N_RUNS) for c in selected for ns in SAMPLE_COUNTS)
    log(f"LIME progress: {n_done}/{total}. " + ("ALL RUNS COMPLETE." if n_done >= total else "Run the same command again to continue."))


if __name__ == "__main__":
    main()
