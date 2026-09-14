"""
LIME Instability Experiment — Round 2 (confidence-filtered)
=============================================================
LegalPro-BERT on CUAD legal clauses: tests whether LIME explanations
are more stable (higher mean pairwise Jaccard@5 across N=5 runs) for
fixed-formula clauses (Governing Law) than compositional clauses
(Liquidated Damages).

This is the exact script structure used to produce the Round 2 results
already discussed:
    Governing Law:      mean Jaccard@5 = 0.071  (n=3 clauses)
    Liquidated Damages: mean Jaccard@5 = 0.056  (n=2 clauses)
i.e. no clean class-level separation was found. This script is meant
to let you REPRODUCE that run end-to-end and/or EXTEND it (more
clauses, per Step 2 of the plan) on your own machine.

--------------------------------------------------------------------
REQUIRED FILES (place these in the same folder, or edit the paths
in CONFIG below) — confirmed present in your LIME_Folder on Desktop:
  1. legalpro_bert_final_trainvaltest_best.pt
       - checkpoint dict with keys 'state_dict' and 'labels'
         (labels = the 41-item ordered list of CUAD clause-type names)
  2. CUAD_v1.json
       - the standard CUAD v1 dataset file
  3. split_record_TRAINVALTEST_PERMANENT.json
       - your saved train/val/test split record, containing
         'test_context_hashes': a list of sha1 hashes identifying
         which contracts belong to the held-out test set (this is
         what keeps this experiment on the same clean test split used
         throughout the paper — do not swap in a different split)

NOTE ON VOCAB: the original session used a custom 'bert_vocab_candidate.txt'
file that isn't present locally. Since the checkpoint's config uses
vocab_size=30522 (the standard bert-base-uncased vocab size), this
script loads the tokenizer directly from HuggingFace instead
(BertTokenizer.from_pretrained("bert-base-uncased")) rather than
requiring that file. This should be equivalent, but if any run
produces obviously garbled tokens or a shape mismatch, that's the
first thing to revisit.

--------------------------------------------------------------------
HOW TO RUN
This is written to be resumable and memory-safe, because a full N=5
run per clause on CPU can be slow and RAM-heavy. Progress is saved to
`lime_progress_round2.json` after every single LIME run, so you can
stop and restart freely without losing work.

    # run a batch of up to 2 LIME runs, then stop
    MAX_RUNS=2 python3 lime_instability_round2.py

    # keep calling this same command until it prints "ALL RUNS COMPLETE."
    MAX_RUNS=2 python3 lime_instability_round2.py

Once complete, run the summary block at the bottom (or just re-run
the script again — it will print the summary automatically once
total_done >= total_needed).

--------------------------------------------------------------------
WHAT TO SEND BACK
Once you see "ALL RUNS COMPLETE." and the final Jaccard@5 summary
table printed, send me:
  1. the full console output of that final run (the summary table)
  2. the `lime_progress_round2.json` file it produces
That's everything needed to document the result and draft the
GitHub README together.
"""

import os
import json
import hashlib
import unicodedata
import re
import itertools
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, AutoTokenizer
from lime.lime_text import LimeTextExplainer

torch.set_num_threads(2)

# ----------------------------------------------------------------------
# CONFIG — edit these paths to match your local files
# ----------------------------------------------------------------------
CHECKPOINT_PATH = "legalpro_bert_final_trainvaltest_best.pt"
CUAD_PATH = "CUAD_v1.json"
SPLIT_RECORD_PATH = "split_record_TRAINVALTEST_PERMANENT.json"
PROGRESS_FILE = "lime_progress_round2.json"

MAX_LEN = 384
N_RUNS = 5          # LIME runs per clause
TOP_K = 5           # top-k features used for Jaccard
NUM_SAMPLES = 150   # LIME perturbation samples per run
MAX_RUNS_THIS_CALL = int(os.environ.get("MAX_RUNS", 2))

# The 5 clauses used in Round 2 (confidence-filtered: only clauses
# where the model's own predicted probability for the gold label
# was >= 0.5, found by scanning the full test set — see note below).
# Format: (contract title prefix, clause label, char_pad window size)
SELECTED = [
    ("NeoformaInc", "Governing Law", 1200),
    ("RevolutionMedicinesInc", "Governing Law", 1200),
    ("HertzGroupRealtyTrust", "Governing Law", 1200),
    ("BOLIVARMININGCORP", "Liquidated Damages", 4000),
    ("TRUENORTHENERGYCORP", "Liquidated Damages", 4000),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize_context(s):
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()


def context_hash(ctx):
    return hashlib.sha1(normalize_context(ctx).encode("utf-8")).hexdigest()


def clause_window(full_text, char_start, char_end, char_pad):
    lo = max(0, char_start - char_pad)
    hi = min(len(full_text), char_end + char_pad)
    return full_text[lo:hi]


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
    """LIME-compatible predict_proba wrapper, batched to avoid CPU OOM."""

    def __init__(self, tokenizer, model, target_idx, batch_size=8, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.model = model
        self.target_idx = target_idx
        self.batch_size = batch_size
        self.max_len = max_len

    def predict_proba(self, texts):
        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(batch, truncation=True, max_length=self.max_len,
                                  padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(enc["input_ids"], enc["attention_mask"])
                probs = torch.sigmoid(logits)[:, self.target_idx].numpy()
            all_probs.append(probs)
            del enc, logits
        p = np.concatenate(all_probs)
        return np.stack([1 - p, p], axis=1)


def main():
    for path in (CHECKPOINT_PATH, CUAD_PATH, SPLIT_RECORD_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required file: {path}\n"
                f"Place it in the same folder as this script, or edit the "
                f"CONFIG paths at the top of the file to point to LIME_Folder."
            )

    log("Loading split record and CUAD...")
    with open(SPLIT_RECORD_PATH) as f:
        split = json.load(f)
    test_hashes = set(split["test_context_hashes"])
    with open(CUAD_PATH) as f:
        cuad = json.load(f)["data"]

    log("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                         num_attention_heads=12, intermediate_size=3072,
                         max_position_embeddings=512, type_vocab_size=2)
    model = LegalProBERT(config)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    labels = ckpt["labels"]
    log("Model ready.")

    # Rebuild the exact clause windows for the 5 selected clauses
    clause_windows = {}
    for contract in cuad:
        ctx = contract["paragraphs"][0]["context"]
        if context_hash(ctx) not in test_hashes:
            continue  # only use held-out test-set contracts
        for qa in contract["paragraphs"][0]["qas"]:
            label = qa["id"].split("__")[-1]
            if label in ("Governing Law", "Liquidated Damages") and not qa["is_impossible"] and qa["answers"]:
                for prefix, want_label, pad in SELECTED:
                    if contract["title"].startswith(prefix) and label == want_label:
                        ans = qa["answers"][0]
                        cs, ce = ans["answer_start"], ans["answer_start"] + len(ans["text"])
                        clause_windows[f"{prefix}|{want_label}"] = clause_window(ctx, cs, ce, pad)

    missing = [f"{p}|{l}" for p, l, _ in SELECTED if f"{p}|{l}" not in clause_windows]
    if missing:
        log(f"WARNING: could not find these clauses in the test set: {missing}")

    # Load or initialize progress
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
    else:
        progress = {key: [] for key in clause_windows}

    explainer = LimeTextExplainer(class_names=["other", "target"])
    runs_done_this_call = 0

    for key, text in clause_windows.items():
        prefix, label = key.split("|")
        target_idx = labels.index(label)
        wrapper = ModelWrapper(tokenizer, model, target_idx)
        while len(progress.get(key, [])) < N_RUNS and runs_done_this_call < MAX_RUNS_THIS_CALL:
            run_idx = len(progress.setdefault(key, []))
            t0 = time.time()
            np.random.seed(3000 + run_idx)
            exp = explainer.explain_instance(text, wrapper.predict_proba, num_features=10,
                                              num_samples=NUM_SAMPLES, labels=(1,))
            top_words = [w for w, _ in exp.as_list(label=1)[:TOP_K]]
            progress[key].append(top_words)
            runs_done_this_call += 1
            log(f"[{key}] run {run_idx + 1}/{N_RUNS} done in {time.time() - t0:.1f}s: {top_words}")
            with open(PROGRESS_FILE, "w") as f:
                json.dump(progress, f, indent=2)
        if runs_done_this_call >= MAX_RUNS_THIS_CALL:
            break

    total_done = sum(len(v) for v in progress.values())
    total_needed = len(clause_windows) * N_RUNS
    log(f"Progress: {total_done}/{total_needed} runs complete.")

    if total_done >= total_needed and total_needed > 0:
        log("ALL RUNS COMPLETE. Computing final Jaccard@5 summary...")
        by_label = {"Governing Law": [], "Liquidated Damages": []}
        for key, runs in progress.items():
            prefix, label = key.split("|")
            pairs = list(itertools.combinations(range(len(runs)), 2))
            jscores = [jaccard(runs[i], runs[j]) for i, j in pairs]
            mean_j = float(np.mean(jscores))
            std_j = float(np.std(jscores))
            by_label[label].append(mean_j)
            print(f"{label:20s} | {prefix:25s} | mean Jaccard@5 = {mean_j:.3f} (std={std_j:.3f})")

        print("\n=== SUMMARY BY CLAUSE TYPE ===")
        for label, means in by_label.items():
            if means:
                print(f"{label:20s}: n={len(means)}  mean={np.mean(means):.3f}  "
                      f"std={np.std(means):.3f}  values={[round(m, 3) for m in means]}")

        with open("lime_final_summary_round2.json", "w") as f:
            json.dump({"by_label_means": by_label}, f, indent=2)
        log("Saved lime_final_summary_round2.json — send this + the console output back.")


if __name__ == "__main__":
    main()
