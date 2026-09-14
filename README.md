# Does LIME Actually Explain BERT? (A Legal-Clause Case Study)

**Status: pilot run complete (5 clauses). Short answer: not really — and not in the way I expected.**

## Why I started this

LIME explains a model's prediction by fitting a simple linear model
around it — basically asking "if I nudge these words, how does the
prediction change?" and reading off which words moved it most. That's a
reasonable thing to do for a linear model. It's a much shakier thing to
do for BERT, where every word's meaning is entangled with every other
word through attention. A linear approximation of something fundamentally
non-linear and interactive seems like it should break somewhere — the
question is where.

My guess going in was that it would break selectively, not everywhere.
Some clause types in legal contracts are basically fixed formulas —
*Governing Law* is almost always some version of "this agreement is
governed by the laws of [State]." A single word or two carries the whole
signal, so I expected LIME to find that reliably every time. Other clause
types, like *Liquidated Damages*, only make sense as a combination — you
need an amount, a trigger condition, and a legal term all showing up
together. There's no single anchor word to latch onto, so I expected
LIME's linear approximation to fall apart there specifically.

That was the hypothesis: LIME should be stable on formulaic clauses and
unstable on compositional ones, because the mismatch between "linear
surrogate" and "attention-based interaction" should bite harder when the
signal is genuinely compositional.

## Method

- **Model:** LegalPro-BERT, a `bert-base-uncased`-initialized multi-label
  classifier fine-tuned on CUAD (41 clause types, sigmoid output,
  classification threshold 0.35).
- **Data:** CUAD v1, restricted to the held-out test split used
  throughout the accompanying paper (`split_record_TRAINVALTEST_PERMANENT.json`),
  to avoid any train/test leakage into this analysis.
- **Clause selection (confidence-filtered):** clauses were scanned for
  model confidence on the gold label; only clauses where the model's own
  predicted probability was ≥0.5 were used, to avoid confounding LIME
  instability with prediction uncertainty near the 0.35 decision threshold.
  (An earlier, unfiltered pass using near-threshold clauses, P≈0.39–0.47,
  produced near-zero Jaccard scores for both clause types and was discarded
  as confounded — see `results/round1_near_threshold.json`.)
- **Windowing:** each clause is scored using a clause-centered window
  (the gold clause span plus surrounding context, padded to fit within
  the model's input length), since gold clauses in CUAD frequently sit
  thousands of tokens into the full contract, far beyond a naive
  from-the-start truncation.
- **Stability metric:** for each clause, LIME is run **N=5** independent
  times (`num_samples=150` perturbations per run). Each run's top-5
  highest-weighted words are recorded. Stability is the **mean pairwise
  Jaccard similarity** of the top-5 word sets across all C(5,2)=10 run
  pairs: `Jaccard(A,B) = |A∩B| / |A∪B|`, ranging 0 (no overlap) to 1
  (identical).

## What actually happened

I ran this twice, in two different environments, on the same 5 clauses
(3 Governing Law, 2 Liquidated Damages), partly to sanity-check the setup
and partly because the first result was surprising enough that I wanted
to see if it held up.

| Clause type | Run 1 mean Jaccard@5 | Run 2 mean Jaccard@5 |
|---|---|---|
| Governing Law (n=3) | 0.071 | 0.015 |
| Liquidated Damages (n=2) | 0.056 | 0.022 |

It didn't go the way I expected. Governing Law — the "should be easy and
stable" clause type — wasn't stable at all. Run the same clause through
LIME five times and you'll typically get five almost entirely different
sets of top words; a Jaccard score under 0.1 means the runs agree on
basically nothing. And Liquidated Damages, if anything, came out no
worse (and in run 2, slightly better) than Governing Law — the opposite
direction from what I'd predicted.

So the class-level story I was hoping to tell — "LIME struggles
specifically with compositional clauses" — just isn't in this data. What
*is* in the data is something more blunt: LIME looks unreliable on this
model **across the board**, regardless of whether the clause type has an
obvious anchor word or not.

One more thing worth flagging honestly: the two runs don't even agree
with each other on the exact numbers (0.071 vs. 0.015 for Governing Law
is a big gap). Some of that is just LIME's own randomness in how it
perturbs text. But it's a little uncomfortable that a "stability" metric
is itself this unstable between runs — if anything, that's more evidence
for the broader point, not less.

**Where I've landed:** the defensible claim here isn't "LIME fails on
compositional legal language specifically." It's "LIME's explanations
for this BERT-based legal classifier are unreliable, full stop" — which
is a less tidy story than I went in wanting, but it's the one the data
actually supports, and it's still a real finding: it says something
about the limits of applying a linear local surrogate to an
attention-based model, just not the specific mechanism I originally
bet on.

## What this doesn't show (yet)

- **The sample is small** — 3 clauses and 2 clauses isn't enough to
  rule the original hypothesis out for good, just enough to say it's
  not showing up here. Scaling to ~10 clauses per class is the obvious
  next step, and I haven't done it yet.
- I've only tested this one pair of clause types. The
  formulaic-vs-compositional idea might hold for a different pairing
  even if it doesn't for this one.
- I ran LIME with 150 perturbation samples per explanation, which is on
  the low side. More samples might quiet down the per-run noise — I
  haven't checked whether that changes the picture, and it costs more
  compute to find out.

## Running it yourself

`lime_instability_round2.py` has everything. You'll need three files
locally that aren't in this repo (see below for why):
- `legalpro_bert_final_trainvaltest_best.pt` — the trained checkpoint
- `CUAD_v1.json` — the CUAD dataset ([get it here](https://www.atticusprojectai.org/cuad))
- `split_record_TRAINVALTEST_PERMANENT.json` — the held-out test-split record

```bash
pip install torch transformers lime scikit-learn numpy
MAX_RUNS=2 python3 lime_instability_round2.py
```

It's resumable — it saves progress after every single LIME run, so if it
times out or you want to stop partway, just run the same command again
and it'll pick up where it left off. Keep going until you see
`ALL RUNS COMPLETE.`.

## Why the data files aren't in here

The checkpoint's a large binary and doesn't belong in git. CUAD has its
own license and is better pulled from the source above. The split record
encodes derived info from the training pipeline. Reach out if you need
any of these for verification.

## Where this fits in the bigger picture

This is one piece of a broader thread I'm working on — NLP
interpretability for legal text:
- Multi-label clause classification (LegalPro-BERT, CUAD, micro-F1 0.72) — dissertation repo
- *"Does Compression Preserve What Classification Needs?"* — submitted to JURIX 2026, on whether compressed contract representations keep the information a classifier actually relies on
