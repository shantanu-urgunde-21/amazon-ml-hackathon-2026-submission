# Business Entity Resolution Pipeline

**Amazon ML Challenge 2026** — match every Source 1 business to all of its Source 2 / Source 3 records.

## How to run

Requirements: Python 3.12, ~8 GB free RAM, ~5 GB free disk. Runs on CPU; an NVIDIA GPU (CUDA 12)
is used automatically for FAISS search and XGBoost training when installed with `GPU=1`.

```bash
# from the repository root
make init                                   # CPU: .venv from requirements.txt
# make init GPU=1                           # NVIDIA GPU: requirements-gpu.txt (faiss-gpu-cu12)
cp code/business_entity_resolution/config.example.toml code/business_entity_resolution/config.toml
make run                                    # = python src/main.py : all stages, data -> blocking -> model -> output
```

`config.toml` → `[gpu] useGpu = true` uses the GPU when one is found and falls back to the CPU
otherwise; the FAISS search is exact either way, so results do not depend on the device.

Paths in `config.toml` are relative to the repository root, so the defaults work as long as the
competition data is in `student_resource/dataset/`. Intermediate results are cached in `cache/`
(safe to delete; everything is regenerated).

Run one stage (earlier stages are loaded from the cache):

```bash
cd code/business_entity_resolution
.venv/bin/python src/main.py --stage prepare      # Indic lexicon + normalized parquet            (~8 min)
.venv/bin/python src/main.py --stage candidates   # FAISS candidates, train + test                 (see timings below)
.venv/bin/python src/main.py --stage train        # XGBoost, 5-fold GroupKFold (GPU if available)
.venv/bin/python src/main.py --stage evaluate     # holdout benchmark -> cache/benchmark.json
.venv/bin/python src/main.py --stage predict      # output/candidate_pairs.tsv + output/matching_results.tsv
```

Validate the submission files (from the repository root):

```bash
code/business_entity_resolution/.venv/bin/python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

## Pipeline

```text
raw TSVs (S1, S2, S3)
  │
  ▼  1. normalization/           per record, parallel over unique strings
  │     • unicode fold, accents, zero-width chars, dotted acronyms (L.L.C. -> llc)
  │     • Indic scripts -> English with a lexicon LEARNED from training links
  │       ("प्राइवेट" -> private, "महाराष्ट्र" -> maharashtra), anyascii fallback
  │     • names: legal forms (US / India / France), honorifics, DBA / formerly / nee aliases,
  │       domains (x.com, "| www.x.com"), phone numbers, OCR digits (Appare1s)
  │     • addresses: state/region codes per country (incl. French departments), abbreviations
  │       (St, Rd, R., Bd ...), house numbers, PO box / PMB / CEDEX, "City of", null tokens
  │
  ▼  2. filters/ + retrieval/    FAISS similarity index (candidate generation)
  │     • char 2-4-gram TF-IDF -> SVD (128 d) vectors for name and for address
  │     • exact FAISS inner-product indexes over S1, per country and state partition
  │     • reverse assignment: each S2/S3 record looks up its nearest S1 records
  │       (every pool record belongs to at most one S1 entity)
  │
  ▼  3. features/                pair features, vectorized with rapidfuzz.cpdist
  │     name / address similarities, slot agreement (+1 / 0 / -1), retrieval
  │     cosines, reverse rank / margin, candidate-set context
  │
  ▼  4. models/                  XGBoost, 5-fold GroupKFold by S1 entity (CUDA if available)
  │
  ▼  5. prediction/              one S1 per pool record, then the entity-level gate
  │     (tau_singleton, tau_match, delta_prob) tuned for macro F0.5
  │
  ▼  output/candidate_pairs.tsv, output/matching_results.tsv
```

## Validation protocol (no leakage)

* Train S1 entities are split once into **fit (80%)** and **holdout (20%)**, by entity. Every
  S2/S3 record belongs to at most one S1 entity, so no true pair crosses the split.
* Everything learned from labels uses fit entities only: the Indic lexicon, the model, and the
  retrieval settings.
* The gate thresholds are tuned on one half of the holdout. The benchmark is reported on the
  other half, which nothing was fitted or tuned on.
* The one-S1-per-record step uses out-of-fold probabilities for trained entities, never
  in-sample ones.
* Embedders and FAISS indexes use no labels. They are fitted the same way on the test set.

## Results

Holdout benchmark (`cache/benchmark.json`): 221,025 training S1 entities that no fitting or tuning
step used (the "report half" of the holdout; see the validation protocol above).

| Metric | Value |
| :--- | ---: |
| **Macro F0.5** | **0.9537** |
| Macro F0.5, India / US | 0.9418 / 0.9617 |
| Macro F0.5 on the gate's tuning half | 0.9540 |
| Candidate recall (true links inside the candidate set) | 94.46% |
| Candidates per S1 entity | 8.4 |
| Pair precision / pair recall of the final matches | 98.78% / 90.38% |
| Singletons correctly predicted empty | 94.5% |
| Gate thresholds (τ_singleton, τ_match, Δ_prob) | 0.65, 0.05, 0.30 |

Test set (`output/`, passes `validate_submission.py --check-ids`):

| Country | S1 entities | Pool records | Candidates per S1 | Matches per S1 |
| :--- | ---: | ---: | ---: | ---: |
| France | 259,452 | 1,434,993 | 11.47 | 3.49 |
| India | 809,986 | 4,717,565 | 10.92 | 3.13 |
| US | 663,106 | 3,817,031 | 9.58 | 3.26 |
| **All** | **1,732,544** | **9,969,589** | **10.49** (18.17M pairs) | 3.23 |

Measured run times (Ryzen 7 7445HS, 12 threads, RTX 2050 4 GB): prepare 6 min; candidates
90 min for train + test (CPU FAISS; the GPU index search is ~25x faster, the char n-gram
embedding stays on the CPU); train 5.6 min (XGBoost on GPU; 17 min for the earlier LightGBM on
CPU); evaluate 10.5 min; predict 10.4 min.

Model comparison on the same holdout half: LightGBM (CPU, 1,000 rounds, coarse gate grid)
0.9525 → XGBoost (GPU, early-stopped up to 3,000 rounds, finer gate grid) **0.9537**.

## Source layout

```text
src/
├── config.py            config.toml -> paths and parameters
├── main.py              CLI over the pipeline stages
├── pipeline.py          stages: prepare, candidates, train, evaluate, predict (cached)
├── dataloader/          TSV loading, fit/holdout split, lexicon building, normalized cache
├── normalization/       lexicons.py, text.py (folding, phonetic key), transliterate.py, normalizer.py
├── retrieval/           embedder.py (char n-gram TF-IDF + SVD), index.py (FAISS, GPU/CPU)
├── filters/             blocking.py: partitioned FAISS reverse assignment, recall evaluation
├── features/            extractor.py: pair features
├── models/              classifier.py: XGBoost + GroupKFold
├── prediction/          gate.py: exclusive assignment, decision gate, vectorized macro F0.5
├── validation/          metrics.py: reference macro F0.5 (competition definition)
└── notebooks/           exploration
```
