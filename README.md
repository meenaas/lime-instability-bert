# Does LIME Explain a BERT Legal-Clause Classifier?

**Status (4 Oct 2026): round 4 complete. The instability I first reported was
an artefact of my setup. With that fixed, LIME is fairly stable, and what it
shows is that the classifier relies on contract type, not on the clause.**

## Summary

| Round | Setup | Result |
|---|---|---|
| 2 | 150 LIME samples; 384-token windows cut from the contract | Jaccard@5 of 0.06-0.07. Looked like LIME was unreliable. |
| 3 | Token-budgeted windows, sample sweep | The model was unsure on almost every window (3 of 64 Governing Law clauses above 0.5). Something was wrong with the input. |
| 4 | The model's real input (first 512 tokens of the contract), sample sweep up to 2,500 | Jaccard@5 of 0.65-0.77. Attributions follow the contract, not the label. |

## What I got wrong in round 2

1. **Too few samples.** I used 150 perturbation samples per explanation.
   LIME's default is 5,000.
2. **Wrong input.** The classifier was trained on the first 512 tokens of each
   contract. I ran LIME on 384-token windows cut from around each clause, which
   the model had never seen. For Liquidated Damages the window was padded by
   4,000 characters, so the clause was probably outside the model's input.

Round 2 therefore measured LIME's noise on unfamiliar input. It did not test
my hypothesis that LIME's linear surrogate breaks on compositional clauses.

## Round 4 method

- **Model:** LegalPro-BERT fine-tuned on CUAD for 41-label clause
  classification (test micro-F1 0.608, macro-F1 0.486, threshold 0.35, on a
  held-out 356/76/77 contract split).
- **Input:** the first 512 tokens of each test contract, as in training.
- **Labels:** License Grant, Governing Law, Effective Date.
- **Items:** 6 test contracts per label where the label is present and the
  model predicts it.
- **LIME:** 5 runs per item at 150, 500, 1,000 and 2,500 samples, each with a
  different seed. Top-5 words by absolute weight.
- **Stability:** mean pairwise Jaccard of the top-5 word sets across the 5 runs.
  Random top-5 lists over the same vocabulary would give about 0.015.

## Results

### 1. Stability rises with sample count

| Samples | License Grant | Governing Law | Effective Date |
|---|---|---|---|
| 150 | 0.10 | 0.06 | 0.06 |
| 500 | 0.49 | 0.45 | 0.39 |
| 1,000 | 0.61 | 0.59 | 0.55 |
| 2,500 | 0.65 | 0.77 | 0.69 |

The 150-sample row reproduces round 2. There is no difference between clause
types.

### 2. Attributions follow the contract, not the label

Taking the words that appear in at least 3 of 5 runs at 2,500 samples:

| Comparison | Pairs | Jaccard |
|---|---|---|
| Same contract, different label | 9 | 0.56 |
| Same label, different contract | 45 | 0.02 |

For one distribution agreement, all three labels return "distributor",
"products", "breach", "patents".

### 3. The top words are not clause words

Share of top-5 words that occur in the gold clause: Governing Law 0.05,
Effective Date 0.01. The words LIME picks describe the kind of agreement and
the parties: "distributor", "reseller", "collaboration", "manufacturing",
"license", "alliance".

This fits where the clauses sit. Governing Law is usually near the end of a
contract, so it is outside the first 512 tokens in every test contract here,
yet the model still predicts it.

### 4. The linear fit is modest

LIME's local R² stays near 0.4 at every sample count.

## What I conclude

- The low stability in round 2 was under-sampling.
- On this model, LIME is stable enough to be informative at 1,000+ samples.
- What it shows is that the classifier appears to use a general impression of
  contract type from the opening text, and moves several labels together.

## What this does not show

- Six contracts per label and three labels. This is a pattern, not a proven
  mechanism.
- LIME removes every occurrence of a word at once, so it cannot show effects
  of position or order.
- Stable attributions are not the same as faithful ones. I have not yet
  checked them against a deletion test or another attribution method.
- I did not go above 2,500 samples.

## Next

- Deletion test: remove LIME's top words and measure the change in prediction.
- Compare with gradient-based attribution and attention-head evidence.
- Repeat on a classifier that sees the whole contract (chunked input).

## Files

| File | Purpose |
|---|---|
| `round4_diagnose_and_lime.py` | Round 4: diagnostics and LIME sweep |
| `lime_round4_summary.csv` | Stability table above |
| `lime_round4_progress.json` | Every run: seed, top words, weights, R² |
| `lime_round4_clauses.json` | The 18 selected items and their input text |
| `lime_instability_round3.py` | Round 3 (superseded) |
| `lime_instability_round2.py`, `lime_*_round2.json` | Round 2 (superseded) |
| `split_record_TRAINVALTEST_PERMANENT.json` | Train/val/test contract hashes |

## Reproducing

Not included: the trained checkpoint (too large for git), CUAD
([Atticus Project](https://www.atticusprojectai.org/cuad)) and the summary
files used alongside it. Contact me for these.

    pip install torch transformers lime scikit-learn pandas numpy
    RUN_LIME=1 REQUIRE_VISIBLE=0 \
      LIME_LABELS="License Grant,Governing Law,Effective Date" \
      SAMPLE_COUNTS=150,500,1000,2500 python3 round4_diagnose_and_lime.py

The run is resumable and takes about 2.5 hours on an Apple Silicon GPU.

## Context

The classifier comes from *"Does Compression Preserve What Classification
Needs?"* (with Dr. Alaa Marshan, submitted to JURIX 2026). That paper does not
use LIME; this repository is separate follow-up work.
