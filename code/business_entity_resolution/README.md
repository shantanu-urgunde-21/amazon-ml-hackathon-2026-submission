# Business Entity Resolution Pipeline

**Amazon ML Challenge 2026**: for every Source 1 (S1) business, find all of its records in Source 2
and Source 3 (S2/S3). This README is the complete guide. It covers how to run everything, how each
stage works, how validation avoids leakage, the results, and where each piece of code lives.

---

## 1. Run it (GPU)

**Needs:** Linux, Python 3.12, an NVIDIA GPU whose driver supports CUDA 12 (`nvidia-smi` works),
about 8 GB of free RAM and 10 GB of free disk. The CUDA libraries come from pip; no system CUDA
toolkit is needed. Tested on a Ryzen 7 7445HS (12 threads) with an RTX 2050 (4 GB).

The competition data must be in `student_resource/dataset/` (`train/` and `test/`).

```bash
# from the repository root
make clean-all                     # start from a fresh venv (faiss-cpu and faiss-gpu cannot share one)
make init GPU=1                    # .venv from requirements-gpu.txt: faiss-gpu-cu12, xgboost, ...
cp code/business_entity_resolution/config.example.toml code/business_entity_resolution/config.toml
make run                           # all stages -> output/candidate_pairs.tsv + output/matching_results.tsv

# check the submission files
code/business_entity_resolution/.venv/bin/python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test --check-ids

make package TEAM=<team_name>      # <team_name>_submission.zip: output/, code/, Documentation_template.md
```

**Is the GPU in use?** Check with

```bash
cd code/business_entity_resolution
.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from src.retrieval import gpu_available; print(gpu_available())"
```

It should print `True`. During training the log also shows `(XGBoost on cuda)`.

**No GPU?** Use `make init` instead of `make init GPU=1` and leave everything else the same. The
code falls back to the CPU on its own. FAISS search is exact on both devices, so the candidates are
identical; only the run time changes.

All intermediate results are cached in `cache/` at the repository root (about 3.5 GB, safe to delete).
A stage whose inputs are already cached is not recomputed. To run one stage:

```bash
cd code/business_entity_resolution
.venv/bin/python src/main.py --stage <prepare|candidates|train|evaluate|predict>
```

---

## 2. Stages

| Stage | What it does | Writes (in `cache/` unless noted) | Time* |
| :--- | :--- | :--- | ---: |
| `prepare` | Split train S1 entities into fit / holdout; learn the Indic lexicon from fit links; normalize all six files | `s1_split.json`, `indic_lexicon.json`, `normalized/*.parquet` (one per source × country) | 6 min |
| `candidates` | Build the FAISS indexes and generate candidate pairs for train and test, per country | `candidates/*.parquet` | 90 min (CPU FAISS); search ~25× faster on GPU |
| `train` | Pair features for 300k fit entities → 5-fold XGBoost | `models.pkl`, `oof_train.parquet` | 5.6 min (GPU) |
| `evaluate` | Score all train pairs; tune the decision gate on half the holdout; benchmark on the other half | `gate.json`, `benchmark.json`, `holdout_report_pairs.parquet` | 10.5 min |
| `predict` | Score the test set, apply the gate, write the submission | `output/candidate_pairs.tsv`, `output/matching_results.tsv` | 10.4 min |

\* measured on the machine above. `candidates` was run with CPU FAISS; on the GPU, only the index
search gets faster, because the char n-gram embedding runs on the CPU.

---

## 3. How it works

### 3.1 Facts about the data the design relies on

* Every S2/S3 record matches **at most one** S1 entity, and links never cross countries.
* An S1 entity has 3.46 matches on average (99th percentile: 8), and 5.6% have none. 26% of pool records match nothing.
* S1 is always Latin script. 41% of India S2/S3 records contain Indic script (Devanagari, Bengali, Tamil,
  Telugu, Kannada, Gujarati, Malayalam, Odia, Gurmukhi). These are mostly English words written in that script.
* The test set adds France (15% of S1), which has no training labels.

### 3.2 Pipeline

```text
raw TSVs (S1, S2, S3)
  │
  ▼  1. NORMALIZATION                       normalization/, dataloader/loader.py
  │     • unicode fold, accents, zero-width chars, dotted acronyms (L.L.C. -> llc), OCR digits (Appare1s)
  │     • Indic words -> English via a lexicon LEARNED from fit-split links
  │       ("प्राइवेट" -> private, "தமிழ்நாடு" -> tamil nadu); anyascii for unknown words
  │     • names: legal forms (US / India / France), honorifics, DBA / formerly / née aliases,
  │       domains (x.com, "| www.x.com", ...COM), phone numbers  ->  name_core, name_alias, legal, ...
  │     • addresses: state / region codes per country (incl. native scripts, French departments),
  │       abbreviations (St, Rd, R., Bd ...), house numbers (C-2-08 = C-208), PO box / PMB / CEDEX, null
  │
  ▼  2. CANDIDATES: FAISS similarity index   retrieval/, filters/blocking.py
  │     • vectors: char 2-4-gram TF-IDF -> SVD (128 d), one for the name, one for the address
  │     • exact FAISS inner-product indexes over S1 records, per country x state partition
  │     • REVERSE ASSIGNMENT: each S2/S3 record looks up its nearest S1 records, and an S1 entity's
  │       candidates are the records that point at it. This is small and adaptive (8.4 per S1).
  │
  ▼  3. FEATURES                             features/extractor.py
  │     36 pair features computed column-wise (rapidfuzz.cpdist): name / address similarities,
  │     agreement (+1 / 0 / -1) on state, house number and legal form, retrieval cosines,
  │     reverse rank and margin, rank / margin inside the S1 entity's candidate set
  │
  ▼  4. MODEL                                models/classifier.py
  │     XGBoost, 5-fold GroupKFold by S1 entity, fold models averaged
  │
  ▼  5. DECISIONS                            prediction/gate.py
  │     • exclusive assignment: each pool record goes to its most probable S1 entity only
  │     • gate: predict nothing if max P < tau_singleton, else pairs with P >= tau_match
  │       and max P - P <= delta_prob (tuned for macro F0.5)
  │
  ▼  output/candidate_pairs.tsv, output/matching_results.tsv
```

### 3.3 Candidate generation in detail

Per country:

1. Two embedders (name, address) are fitted on 200k of that country's pool records. They use no labels.
2. S1 records are indexed in exact FAISS flat indexes, one per state partition and per view. True
   matches agree on state in more than 98.8% of links. WA and DC share a partition, and so do Andhra
   Pradesh and Telangana, because the data mixes them up.
3. The pool is streamed through the indexes in batches of 500k. Each record searches its state's
   partition; a record without a state searches all partitions of its country. Its nearest S1 records
   by name and by address are re-scored with `0.4·cos(name) + 0.6·cos(address)` (the name cosine
   alone when an address is empty). A pair is kept if the S1 record is:
   * the best match by name alone or by address alone, or
   * the best match by the combined score, or
   * in the top 3 and within 0.05 of the best.

   A record with no address keeps up to 5 by name, within 0.10 of the best.
4. At most 30 candidates are kept per S1 entity.

### 3.4 Validation protocol (no leakage)

* Train S1 entities are split once into **fit (80%)** and **holdout (20%)**, by entity (`cache/s1_split.json`).
  Every S2/S3 record belongs to at most one S1 entity, so no true pair crosses the split.
* Everything learned from labels uses fit entities only: the Indic lexicon, XGBoost (GroupKFold by S1
  entity inside fit), and the retrieval settings.
* The gate thresholds are tuned on one half of the holdout. The benchmark is reported on the other half,
  which nothing was fitted or tuned on.
* The exclusive-assignment step uses out-of-fold probabilities for the trained entities, never in-sample ones.
* The embedders and FAISS indexes use no labels. They are fitted the same way on the test set.

### 3.5 Engineering constraints

* **Scale:** 25M records in total. There are no pairwise loops; candidates come from indexes and
  features are computed column-wise.
* **Memory (~8 GB):** data is cached as one parquet file per (source, country), the pool is streamed,
  and features are built and scored in blocks of 150k entities.
* **Open country set:** every stage loops over the country labels it finds. France uses the same code path.
* **Competition rules:** no external data or APIs, and no pretrained models. The matching model is
  XGBoost (Apache-2.0); FAISS is MIT.

---

## 4. Configuration (`config.toml`)

Paths are relative to the repository root, so the file works wherever the repository is cloned.
`config.example.toml` holds the defaults.

| Section / key | Default | Meaning |
| :--- | :--- | :--- |
| `datasetPaths.*` | `student_resource/dataset/...` | train / test folders, ground truth, validator script |
| `outputPaths.outputDir`, `cacheDir` | `output`, `cache` | submission files, intermediate results |
| `gpu.useGpu` | `true` | use the GPU for FAISS and XGBoost when one is available |
| `retrieval.embedDim` | 128 | SVD dimensions per view |
| `retrieval.nameWeight` | 0.4 | weight of the name cosine in the combined score |
| `retrieval.revK`, `revMargin` | 3, 0.05 | extra S1 records kept per pool record, and their margin |
| `retrieval.searchChunk` | 500000 | pool records per batch (memory bound) |
| `params.maxCandidatesPerS1` | 30 | hard cap per S1 entity |
| `params.holdoutFrac` | 0.2 | share of train S1 entities held out for the benchmark |
| `params.trainEntities` | 300000 | fit entities whose pairs train the model |
| `params.nFolds`, `randomSeed` | 5, 42 | cross-validation folds, seed |

---

## 5. Results

**Holdout benchmark** (`cache/benchmark.json`) on 221,025 training S1 entities that no fitting or
tuning step used:

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

**Test set** (`output/`, passes `validate_submission.py --check-ids`):

| Country | S1 entities | Pool records | Candidates per S1 | Matches per S1 |
| :--- | ---: | ---: | ---: | ---: |
| France | 259,452 | 1,434,993 | 11.47 | 3.49 |
| India | 809,986 | 4,717,565 | 10.92 | 3.13 |
| US | 663,106 | 3,817,031 | 9.58 | 3.26 |
| **All** | **1,732,544** | **9,969,589** | **10.49** (18.17M pairs) | 3.23 |

**Model comparison** (same holdout half): LightGBM on CPU scored 0.9525 (8.5 min training).
**XGBoost on GPU scored 0.9537** (5.6 min).

---

## 6. Code map

```text
src/
├── main.py              CLI: python src/main.py [--stage NAME]
├── pipeline.py          the five stages and their caching
├── config.py            config.toml -> paths and parameters
├── dataloader/          loader.py: TSV loading, fit/holdout split, lexicon building, normalized parquet cache
├── normalization/       lexicons.py (legal forms, states, abbreviations), text.py (folding, phonetic key),
│                        transliterate.py (lexicon learning), normalizer.py (name / address parsing)
├── retrieval/           embedder.py (char n-gram TF-IDF + SVD), index.py (FAISS / PyTorch CUDA flat index),
│                        projector.py (supervised residual metric projector)
├── filters/             blocking.py: partitioned FAISS reverse assignment, blocking recall
├── features/            extractor.py: the 36 pair features
├── models/              classifier.py: XGBoost + GroupKFold, prediction, feature importance
├── prediction/          gate.py: exclusive assignment, decision gate, vectorized macro F0.5
├── validation/          metrics.py: reference macro F0.5 (competition definition) + self-test
└── notebooks/           exploration
```

| Module | Key functions |
| :--- | :--- |
| `dataloader/loader.py` | `load_source()`, `load_ground_truth()`, `split_s1_ids()`, `build_lexicon()`, `build_normalized()` |
| `normalization/` | `normalize_records()`, `normalize_name()`, `normalize_address()`, `fold()`, `phonetic_key()`, `learn_lexicon()` |
| `retrieval/` | `CharNgramEmbedder`, `name_text()`, `addr_text()`, `flat_index()`, `gpu_available()`, `MetricResidualProjector`, `train_metric_projector()` |
| `filters/blocking.py` | `generate_candidates()`, `PartitionedIndex`, `evaluate_blocking_recall()` |
| `features/extractor.py` | `build_features()`, `FEATURE_NAMES` |
| `models/classifier.py` | `cross_validate_model()`, `predict()`, `train_model()`, `feature_importance()` |
| `prediction/gate.py` | `exclusive_assignment()`, `select_pairs()`, `macro_fbeta_pairs()`, `optimize_decision_thresholds()` |
| `validation/metrics.py` | `compute_s1_fbeta()`, `macro_fbeta()`, `self_test()` (`python -m src.validation.metrics`) |
