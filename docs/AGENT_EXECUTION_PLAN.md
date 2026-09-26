# Pipeline Architecture & Operational Playbook

**Amazon ML Challenge 2026 — Business Entity Resolution**

---

## 1. Engineering Guardrails

1. **No quadratic loops.** The train pool has 10.3M records and the test pool 10.0M. Candidates come from partitioned FAISS indexes; features are computed column-wise (`rapidfuzz.process.cpdist`).
2. **GPU optional.** `useGpu = true` + `requirements-gpu.txt`: FAISS flat indexes and XGBoost run on CUDA; exact search, so results match the CPU run.
3. **Memory budget ~8 GB.** Data is cached as one parquet file per (source, country). Candidate generation streams the pool in batches.
4. **Format strictness.** TSV with `sep="\t"`; one row per S1 entity; matches ⊆ candidates.
5. **Competition rules.** No external data or APIs. The only learned resources come from the training files.
6. **Open country set.** Every stage loops over the country labels found in the data. France (test only) is handled by the same code path.
7. **No leakage.** See section 4.

---

## 2. Data Flow

```text
raw TSVs (S1, S2, S3)
  │  dataloader/loader.py        build_normalized(): chunked TSV read -> normalize -> parquet per country
  ▼  normalization/              normalize_records(), learn_lexicon()
  │  filters/blocking.py         generate_candidates(): FAISS reverse assignment per country / state
  ▼  retrieval/                  CharNgramEmbedder (TF-IDF char 2-4-grams -> SVD), FAISS IndexFlatIP
  │  features/extractor.py       build_features(): pair features
  ▼  models/classifier.py        cross_validate_model(), predict()   (XGBoost)
  │  prediction/gate.py          exclusive_assignment(), optimize_decision_thresholds(), select_pairs()
  ▼  output/candidate_pairs.tsv + output/matching_results.tsv
```

`src/pipeline.py` wires these into cached stages: `prepare`, `candidates`, `train`, `evaluate`, `predict`.

---

## 3. Module Contracts

| Module | Responsibility | Key interfaces |
| :--- | :--- | :--- |
| `src/config.py` | `config.toml` → paths and parameters (paths relative to repo root) | `cfg`, `TRAIN_SOURCE1_PATH`, `TEST_SOURCE1_PATH`, `CACHE_DIR`, `REV_K`, `NAME_WEIGHT` |
| `src/dataloader/loader.py` | TSV I/O, fit/holdout split, lexicon building, normalized cache | `load_source()`, `load_ground_truth()`, `split_s1_ids()`, `build_lexicon()`, `build_normalized()` |
| `src/normalization/` | Folding, Indic transliteration, name / address parsing | `normalize_records()`, `normalize_name()`, `normalize_address()`, `fold()`, `phonetic_key()`, `learn_lexicon()` |
| `src/retrieval/` | Dense vectors and FAISS helpers (GPU when available) | `CharNgramEmbedder`, `name_text()`, `addr_text()`, `flat_index()`, `gpu_available()` |
| `src/filters/blocking.py` | Candidate generation and blocking metrics | `generate_candidates()`, `PartitionedIndex`, `evaluate_blocking_recall()` |
| `src/features/extractor.py` | Pair features | `build_features()`, `FEATURE_NAMES` |
| `src/models/classifier.py` | XGBoost (CUDA when available) with GroupKFold by S1 entity | `cross_validate_model()`, `predict()`, `train_model()`, `feature_importance()` |
| `src/prediction/gate.py` | One S1 per pool record, decision gate, vectorized macro F0.5 | `exclusive_assignment()`, `select_pairs()`, `macro_fbeta_pairs()`, `optimize_decision_thresholds()` |
| `src/validation/metrics.py` | Reference macro F0.5 (competition definition) | `compute_s1_fbeta()`, `macro_fbeta()`, `self_test()` |

---

## 4. Validation Protocol (no leakage)

* Train S1 entities are split once into **fit (80%)** and **holdout (20%)**, by entity (`cache/s1_split.json`). Every S2/S3 record belongs to at most one S1 entity, so no true pair crosses the split.
* Everything learned from labels uses fit entities only: the Indic lexicon, XGBoost (GroupKFold by S1 entity inside fit), and the retrieval settings.
* The gate is tuned on one half of the holdout. The benchmark (`cache/benchmark.json`) is reported on the other half.
* The one-S1-per-record step uses out-of-fold probabilities for trained entities.
* Embedders and FAISS indexes use no labels and are fitted the same way on the test set.

---

## 5. Commands

```bash
# from code/business_entity_resolution/
.venv/bin/python src/main.py                     # all stages
.venv/bin/python src/main.py --stage evaluate    # a single stage (uses cached earlier stages)
.venv/bin/python -m src.validation.metrics       # metric self-test

# from the repository root: official format validator
code/business_entity_resolution/.venv/bin/python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
