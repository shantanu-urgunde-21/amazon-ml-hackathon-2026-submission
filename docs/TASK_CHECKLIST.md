# Task Tracker & Milestone Checklist

**Amazon ML Challenge 2026 — Business Entity Resolution**
**Status (2026-09-26):** all phases implemented and run end to end on the full data. Holdout macro F0.5 = **0.9537** (see Phase 5).

> The previous version of this checklist reported 0.9421 from `--sample 5000` runs. Those runs could not be reproduced, and that sample mode contained almost no true matches. Every number below is measured by the current code on the full data.

---

## 📊 Phase Progress Summary

| Phase | Description | Status |
| :--- | :--- | :---: |
| **Phase 0** | Setup, environment, dataset placement | 🟢 Done |
| **Phase 1** | End-to-end pipeline on the full data | 🟢 Done |
| **Phase 2** | Normalization: multilingual, slots, aliases, domains | 🟢 Done |
| **Phase 3** | Precision / recall fixes: contradictions, exclusivity, gate | 🟢 Done |
| **Phase 4** | FAISS similarity-index candidate generation | 🟢 Done |
| **Phase 5** | Test inference, validation, packaging | 🟢 Done |

---

## 📌 Phase Breakdown

### Phase 0: Setup

- [x] `config.toml` (copy of `config.example.toml`), with paths relative to the repo root, so it works on any machine.
- [x] `requirements.txt` pins every direct dependency. A fresh venv installs from it and imports the whole pipeline.
- [x] `make init` / `make run` / `make package`.

### Phase 1: End-to-End Pipeline

- [x] XGBoost (CUDA when available) and FAISS GPU indexes, with automatic CPU fallback (`make init GPU=1`).
- [x] Stages `prepare → candidates → train → evaluate → predict`, each cached (`src/pipeline.py`, `src/main.py`).
- [x] Runs on all 25M records within ~8 GB RAM: data cached per (source, country), pool streamed in batches.
- [x] Vectorized macro F0.5 matches the reference metric (`validation/metrics.py`) exactly.
- [x] Leakage-free protocol: fit/holdout split by S1 entity, gate tuned on one holdout half, reported on the other.

### Phase 2: Normalization

- [x] Indic → English lexicon learned from fit-split links (1,334 words; 84% of distinct test Indic words).
- [x] Country-aware state codes (US, India incl. native scripts, French regions + departments).
- [x] Legal forms (US / India / France), honorifics, DBA / formerly / née aliases, domains, phone numbers, OCR digits.
- [x] Address abbreviations, house numbers (`C-2-08` = `C-208`), PO box / PMB / CEDEX, `City of`, `null`.

### Phase 3: Precision & Recall

- [x] Slot agreement features (+1 / 0 / -1): state, house number, legal form.
- [x] Exclusive assignment: each S2/S3 record goes to at most one S1 entity.
- [x] Entity-level gate (τ_singleton, τ_match, Δ_prob) tuned for macro F0.5.

### Phase 4: FAISS Similarity Index

- [x] Char 2-4-gram TF-IDF → SVD vectors (name, address), exact FAISS IndexFlatIP per country × state partition.
- [x] Reverse assignment: pool records look up their nearest S1 records → small, adaptive candidate sets.
- [x] Candidate recall 94.5% at 8.4 (test: 10.5) candidates per S1 entity (holdout).

### Phase 5: Submission

- [x] Full test inference, including France (unseen in training).
- [x] `outputs/*.tsv` pass `validate_submission.py` (see README → Results).
- [x] `make package TEAM=<name>` builds the submission zip.
