# Amazon ML Challenge 2026 — Task Tracker & Checklist
**Project:** Business Entity Resolution (Source 1 $\rightarrow$ Source 2 & Source 3)  
**Target Metric:** Macro-averaged $F_{0.5}$ (Precision-weighted, singleton-sensitive)  
**Last Updated:** September 25, 2026  

---

## 📊 Overall Progress Summary

| Phase | Description | Priority | Completion | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Phase 0** | Setup, Environment & Data Diagnostics | Essential | 100% | 🟢 Completed |
| **Phase 1** | v0.1 End-to-End Baseline (Recall + Model + Scorer) | Essential | 100% | 🟢 Completed |
| **Phase 2** | High-Yield Enhancements (Slots, Contradictions, Group Margin) | High | 25% | 🟡 In Progress |
| **Phase 3** | Advanced Features (Country Suffixes, S2↔S3 Triangulation) | Medium | 0% | ⚪ Pending |
| **Phase 4** | Experimental (FAISS / Dense Embedding Retrieval) | Experimental | 0% | ⚪ Pending |
| **Phase 5** | Final Verification, Package & Documentation | Essential | 15% | ⚪ Pending |

**Overall Project Completion:** `~55%`

---

## 📌 Phase 0: Setup, Environment & Data Inspection
*Goal: Ensure runnable environment, understand dataset distributions, and prevent data leakage.*

- [x] **0.1 Setup workspace & inspect submission requirements** (100%)
  - Verified folder structure, constraints, and problem statement PDFs.
- [x] **0.2 Python virtual environment & core dependencies** (100%)
  - Base dependencies and ML libraries installed in active venv (`lightgbm`, `rapidfuzz`, `scikit-learn`, `pandas`, `numpy`).
- [x] **0.3 Dataset placement & directory linking** (100%)
  - Centralized path resolution in `src/config.py` pointing to `train/` and `test/` datasets.
- [x] **0.4 Data exploratory diagnostics (`data_diagnostics.py`)** (100%)
  - Streaming ground truth loaded (2,206,821 S1 entities). Verified column headers and format.

---

## 🚀 Phase 1: v0.1 Fast End-to-End Baseline
*Goal: Build a working pipeline from raw TSV to submission-validated outputs within 8–12 hours to establish a benchmark $F_{0.5}$ score.*

### 1.1 Ground Truth & Evaluation Harness
- [x] **1.1.1 TSV Loaders & Data Utilities (`src/dataloader/loader.py`)** (100%)
  - Streaming tab-separated file loader and fast ground truth mapping dictionary.
- [x] **1.1.2 Official Metric Implementation (`src/metrics.py`)** (100%)
  - Exact competition macro-averaged $F_{0.5}$ metric with singleton credit (1.0) and false merge penalty (0.0). Verified against problem statement PDF test cases.
- [x] **1.1.3 Leak-Free Validation Splitter (`src/validation/splitters.py`)** (100%)
  - 5-Fold `GroupKFold` on `source1_entity_id`.

### 1.2 Baseline Normalization (`src/normalization/normalizer.py`)
- [x] **1.2.1 Text Cleaning & Tokenization** (100%)
  - NFKD unicode normalization, lowercasing, punctuation stripping, whitespace collapse, token extraction, and numeric token parsing.

### 1.3 Baseline Candidate Blocking (`src/filters/blocking.py`)
- [x] **1.3.1 Rare Name Token Index** (100%)
  - Inverted index with frequency bucket caps (`MAX_TOKEN_BUCKET_SIZE=1500`) and corporate suffix suppression.
- [x] **1.3.2 Character n-gram Index** (100%)
  - Character 3-gram indexing for spelling errors and transliteration tolerance.
- [x] **1.3.3 Address / Numeric Index** (100%)
  - Numeric address tokens (PIN/ZIP, building/plot numbers) indexed with bucket limits.
- [x] **1.3.4 Candidate Union & Recall Assessment** (100%)
  - Candidate deduplication, country filtering, candidate capping, and recall evaluation function.

### 1.4 Baseline Pairwise Features (`src/features/extractor.py`)
- [x] **1.4.1 Name Similarity Features** (100%)
  - Levenshtein ratio, token-sort ratio, token-set ratio, Jaro-Winkler, exact match, prefix match, length difference.
- [x] **1.4.2 Address Similarity Features** (100%)
  - Address token-set, token-sort, Levenshtein, exact match, numeric token overlap count and Jaccard score.
- [x] **1.4.3 Basic Metadata Features** (100%)
  - Country match equality indicator and source indicator (`cand_is_s2`).

### 1.5 Baseline Model & Entity-Level Decision Gate (`src/models/`, `src/prediction/`, `src/main.py`)
- [x] **1.5.1 LightGBM Binary Classifier (`src/models/classifier.py`)** (100%)
  - GroupKFold LightGBM binary classifier training with early stopping.
- [x] **1.5.2 Threshold Grid Search for Macro-$F_{0.5}$ (`src/prediction/gate.py`)** (100%)
  - Grid search over $(\tau_{\text{singleton}}, \tau_{\text{match}}, \Delta_{\text{prob}})$ directly maximizing macro-$F_{0.5}$.
- [x] **1.5.3 Generate Baseline Outputs & Submission Verification (`src/main.py`)** (100%)
  - End-to-end orchestration implemented in `src/main.py`. Verified on sample run (114,073 pairs scored in 10.9s). Output formatting verified with `validate_submission.py`.

---

## 🔧 Phase 2: High-Yield Enhancements (Tier 2)
*Goal: Address cross-script mismatches, the precision penalty ($F_{0.5}$), and structured inconsistencies.*

- [x] **2.0 Cross-Script Transliteration & Phonetic Skeletons** (100%)
  - Integrated `unidecode` + schwa correction (`kamala` -> `kamal`) into `src/normalization/normalizer.py`.
  - Added phonetic skeleton indexing (`sr`, `blj`) in `src/filters/blocking.py`.
  - Added `is_cross_script` and `phonetic_jaccard` features in `src/features/extractor.py`.
- [ ] **2.1 Structured Address Slot Extraction** (0%)
  - Extract postal codes (5-6 digits), house/building numbers, sub-units (Apt/Suite/Plot/MIDC).
- [ ] **2.2 Ternary Contradiction Logic (+1 / 0 / -1)** (0%)
  - Postal code state: `+1` (agree), `0` (missing in either), `-1` (conflicting).
  - Building/plot number state: `+1` (agree), `0` (missing), `-1` (conflicting).
  - Unit/suite state: `+1` (agree), `0` (missing), `-1` (conflicting).
- [ ] **2.3 Pre-GBDT Group-Relative Context Features** (0%)
  - Margin features: $P_{\text{sim}}(i, j) - \max_{k \ne j} P_{\text{sim}}(i, k)$.
  - Candidate rank by primary name similarity.
  - Candidate bucket size per S1 entity.
  - Density of matching postal codes among candidates.
- [ ] **2.4 Re-train & Benchmark** (0%)
  - Evaluate gain over Phase 1 baseline on 5-fold CV.

---

## 🧠 Phase 3: Advanced Entity Resolution (Tier 3)
*Goal: Country-agnostic robustness and cross-source corroboration.*

- [ ] **3.1 Corpus-Adaptive Legal Suffix Detection** (0%)
  - Compute country-level token position distribution and IDF.
  - Strip suffix from `core_brand` so "Pvt Ltd" / "LLC" does not falsely inflate similarity.
- [ ] **3.2 S2 $\leftrightarrow$ S3 Triangulation / Bridge Features** (0%)
  - Measure pairwise agreement between S2 and S3 candidates within the same S1 pool.
  - Create bridge corroborate feature for missing address fields in S1.
- [ ] **3.3 Leave-One-Country-Out (LOCO) Robustness Check** (0%)
  - Train US $\rightarrow$ Validate India and vice versa to assess transferability to France.

---

## 🔬 Phase 4: Dense Embedding Experiment (Optional / Experimental)
*Goal: Check if dense neural retrieval improves blocking recall or ranking.*

- [ ] **4.1 SentenceTransformer / MiniLM Embedding Extraction** (0%)
  - Embed business name + address strings (Apache 2.0 / MIT models $\le 8\text{B}$).
- [ ] **4.2 FAISS Nearest Neighbor Search** (0%)
  - Generate supplemental candidates for S1.
- [ ] **4.3 Cost vs. Benefit Analysis** (0%)
  - Did candidate recall increase significantly?
  - Did validation $F_{0.5}$ improve? If gain $< 0.005$ or latency is too high, drop.

---

## 📦 Phase 5: Final Submission Package & Verification
*Goal: Ensure flawless compliance with all hackathon rules.*

- [ ] **5.1 Full Test Set Inference** (0%)
  - Run full pipeline on `dataset/test/` (including all France records).
  - Verify every single test S1 appears exactly once in both output files.
  - Verify `matching_results.tsv` matches are strict subsets of `candidate_pairs.tsv`.
- [ ] **5.2 Run Official Validator** (0%)
  - Execute `python utils/validate_submission.py` and ensure `PASS (exit 0)`.
- [ ] **5.3 Reproducibility & Code Cleanup** (0%)
  - Code organized cleanly under `code/business_entity_resolution/src/`.
  - Pinned `requirements.txt` with all needed libraries.
  - Clear `code/business_entity_resolution/README.md` with step-by-step reproduction instructions.
- [ ] **5.4 Methodology Write-up** (0%)
  - Complete `Documentation_template.md` (methodology, blocking strategy, model architecture, experiments).
- [ ] **5.5 Build Final Submission ZIP** (0%)
  - Verify folder structure matches required zip layout.
  - Check total file size and submission limits (max 5 submissions/day).

---

## 📝 Team Action Items & Ownership

| Task ID | Item | Assignee | Target Date | Status |
| :--- | :--- | :--- | :--- | :--- |
| `0.4` | Data diagnostics script & summary | Shantanu | Day 1 | 🟢 Completed |
| `1.1` | TSV loader + exact $F_{0.5}$ scorer | Shantanu | Day 1 | 🟢 Completed |
| `1.2-1.3` | Basic normalizer & indexed blocking | Shantanu | Day 1 | 🟢 Completed |
| `1.4-1.5` | Baseline LightGBM & OOF threshold tuning | Shantanu | Day 1 | 🟢 Completed |
| `1.5.3` | Baseline test inference & validation pass | Shantanu | Day 1 | 🟡 In Verification |
| `2.1-2.3` | Address slots & contradiction logic | TBD | Day 2 | ⚪ Backlog |
| `5.2-5.5` | Submission validation & documentation | TBD | Day 3 | ⚪ Backlog |
