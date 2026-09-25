# Agent Execution Playbook: Phase 1 (v0.1 End-to-End Baseline)
**Target:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Goal:** Deliver a fully tested, compliant baseline producing `matching_results.tsv` and `candidate_pairs.tsv` that passes `validate_submission.py` with a benchmark macro-$F_{0.5}$ score.  
**Location:** Placed in repo root for automated execution.

---

## ⚠️ Non-Negotiable Engineering Guardrails
1. **Data Scale Warning:** Training and test datasets exceed **2.5 GB** (~1.7M test entities). **NEVER write quadratic loops ($O(N \times M)$) or un-indexed comparisons.** Everything must pass through indexed blocking.
2. **Prototype vs Full Mode:** Always implement a `--sample <N>` flag (defaulting to 20,000 S1 entities for rapid development/debugging) before running on full datasets.
3. **Strict Tab-Separation:** Always use `sep="\t"` when reading and writing TSVs. Never use default comma separation.
4. **No External Lookups:** Absolutely no external web requests, geocoding APIs, or external DBs (disqualification rule).
5. **Exact Metric:** Singletons (S1 with no matches) count! An empty prediction for an empty ground truth is a score of `1.0`. A false match on a singleton is `0.0`.

---

## 🗂️ Target File Structure
All code must reside strictly within:
```
d:/programming/amazon ml challange/amazon-ml-hackathon-2026-submission/
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt
│       ├── README.md
│       └── src/
│           ├── __init__.py
│           ├── config.py           # Paths, constants, thresholds, flags
│           ├── metrics.py          # Macro-averaged F0.5 with singleton logic
│           ├── data_utils.py       # Robust TSV loaders & S1 ground truth parser
│           ├── normalize.py        # Unicode, punctuation, whitespace & tokens
│           ├── blocking.py         # Inverted index blocking with bucket caps
│           ├── features.py         # Rapid pairwise string & address similarity
│           ├── model.py            # LightGBM binary classifier + GroupKFold
│           ├── decision.py         # OOF threshold search (tau_singleton, tau_match)
│           └── main.py             # End-to-end pipeline CLI (train -> predict -> write output)
├── output/
│   ├── matching_results.tsv        # Scored leaderboard submission
│   └── candidate_pairs.tsv         # Model candidate set (matches must be subset)
├── TASK_CHECKLIST.md
└── AGENT_EXECUTION_PLAN.md
```

---

## 📋 Step-by-Step Execution Sequence

### STEP 1: Environment & Dependency Setup
* **Target File:** `code/business_entity_resolution/requirements.txt`
* **Action:** Ensure high-performance libraries are installed:
  * `lightgbm>=4.0.0`
  * `rapidfuzz>=3.0.0` (critical for C++ accelerated string similarity)
  * `scikit-learn>=1.3.0`
  * `pandas>=2.0.0`
  * `numpy>=1.24.0`
* **Execution Command:**
  ```powershell
  pip install lightgbm rapidfuzz scikit-learn
  ```
* **Acceptance Criteria:** `python -c "import lightgbm, rapidfuzz, sklearn, pandas; print('DEPS OK')"` exits with code 0.

---

### STEP 2: Configuration Module
* **Target File:** `code/business_entity_resolution/src/config.py`
* **Responsibility:** Centralize all file paths and tunable constants so no paths are hardcoded in logic files.
* **Code Requirements:**
  * Resolve dataset paths pointing to `student_resource/dataset/` or workspace `dataset/`.
  * Define default hyperparams:
    * `MAX_CANDIDATES_PER_S1 = 50`
    * `MAX_TOKEN_BUCKET_SIZE = 1000`
    * `MIN_NAME_TOKEN_LEN = 3`
    * `NGRAM_SIZE = 3`
    * `N_FOLDS = 5`
    * `BETA = 0.5`
* **Verification:** `python -c "import src.config as cfg; print(cfg.TRAIN_S1_PATH)"` runs with no errors.

---

### STEP 3: Evaluation Metric Engine
* **Target File:** `code/business_entity_resolution/src/metrics.py`
* **Responsibility:** Implement exact macro-$F_{0.5}$ scoring per S1 entity.
* **Logic:**
  ```python
  def compute_s1_fbeta(pred_ids: set, true_ids: set, beta: float = 0.5) -> float:
      # Singleton logic:
      if len(true_ids) == 0:
          return 1.0 if len(pred_ids) == 0 else 0.0
      if len(pred_ids) == 0:
          return 0.0
      tp = len(pred_ids & true_ids)
      precision = tp / len(pred_ids)
      recall = tp / len(true_ids)
      if precision + recall == 0:
          return 0.0
      b2 = beta ** 2  # 0.25
      return (1 + b2) * (precision * recall) / (b2 * precision + recall)

  def macro_f05(ground_truth: dict[str, set], predictions: dict[str, set]) -> float:
      scores = [compute_s1_fbeta(predictions.get(s1, set()), true_set)
                for s1, true_set in ground_truth.items()]
      return float(np.mean(scores))
  ```
* **Acceptance Test:** Test with problem statement PDF example:
  * `pred = {"S2-00047", "S2-00193", "S3-00812"}`
  * `true = {"S2-00047", "S3-00812"}`
  * Must output `0.7142857...` ($\approx 0.714$).
  * Test singleton: `true=set()`, `pred=set()` $\rightarrow 1.0$; `pred={"S2-1"}` $\rightarrow 0.0$.

---

### STEP 4: Data Utilities & TSV Parsing
* **Target File:** `code/business_entity_resolution/src/data_utils.py`
* **Responsibility:** Clean loading and validation of S1, S2, S3, and ground truth.
* **Code Requirements:**
  * `load_source(path, sample_n=None) -> pd.DataFrame`: Reads `[entity_id, business_name, business_address, country]`. Handles nulls/empty strings gracefully (`fillna("")`).
  * `load_ground_truth(path) -> dict[str, set]`: Parses `source1_entity_id` and comma-separated `matched_entity_ids`.
  * `get_group_kfold_splits(s1_ids, n_splits=5, seed=42)`: Returns generator of `(train_idx, val_idx)` strictly split by `source1_entity_id`.

---

### STEP 5: Baseline Normalization Engine
* **Target File:** `code/business_entity_resolution/src/normalize.py`
* **Responsibility:** Standardize text while preserving raw fields for downstream feature extraction.
* **Transformations:**
  * Lowercase conversion.
  * Unicode NFKD normalization (replaces accents/ligatures).
  * Replace punctuation characters with spaces (preserve alphanumeric characters).
  * Collapse multiple whitespace into a single space.
  * Extract whitespace tokens.
  * Extract numeric tokens (digits with length $\ge 2$, e.g., PINs and street numbers).
* **Output:** Adds columns `norm_name`, `name_tokens`, `norm_address`, `address_tokens`, `numeric_tokens`.

---

### STEP 6: Scalable Inverted Index Blocking
* **Target File:** `code/business_entity_resolution/src/blocking.py`
* **Responsibility:** Retrieve candidates for S1 entities from S2 & S3 with high recall and capped volume.
* **Algorithm:**
  1. Build inverted indexes over combined pool of S2 and S3:
     * **Index A (Rare Name Tokens):** Index name tokens that appear in fewer than `MAX_TOKEN_BUCKET_SIZE` records and have length $\ge 3$.
     * **Index B (Character 3-grams):** Index 3-grams of normalized name for sub-string and typo tolerance.
     * **Index C (Numeric / Postal Address Tokens):** Index extracted numeric tokens from addresses (capped at 500 records per number).
  2. For each S1 entity:
     * Query indexes and union candidate record IDs.
     * Apply country filtering: candidates must either share the same country or handle open sets.
     * Cap total candidates per S1 entity at `MAX_CANDIDATES_PER_S1` (e.g. 50-100) prioritized by frequency of index hits.
  3. **Candidate Recall Evaluation:** Compute:
     $$\text{Recall} = \frac{\sum_{i} |\text{true\_matches}_i \cap \text{candidates}_i|}{\sum_i |\text{true\_matches}_i|}$$
     Must achieve $\ge 88\text{–}92\%$ candidate recall.

---

### STEP 7: Fast Pairwise Feature Engineering
* **Target File:** `code/business_entity_resolution/src/features.py`
* **Responsibility:** Compute 15–25 pairwise features for candidate pairs using C-optimized `rapidfuzz`.
* **Feature Set:**
  * **Name Similarities:**
    * `levenshtein_ratio(name1, name2)`
    * `jaro_winkler(name1, name2)`
    * `token_set_ratio(name1, name2)`
    * `token_sort_ratio(name1, name2)`
    * Name length ratio and absolute character difference.
    * Exact normalized match boolean (`1` or `0`).
  * **Address Similarities:**
    * `token_set_ratio(addr1, addr2)`
    * Jaccard overlap of address tokens.
    * Character Levenshtein ratio of address.
    * Numeric token intersection count and numeric Jaccard score.
  * **Metadata Features:**
    * Country equality boolean (`1` if same, `0` otherwise).
    * Candidate source indicator (`1` if S2, `0` if S3).

---

### STEP 8: LightGBM Model & GroupKFold Training
* **Target File:** `code/business_entity_resolution/src/model.py`
* **Responsibility:** Train binary classification model $P(\text{match} \mid \text{pair})$.
* **Specification:**
  * Objective: `binary`.
  * Metric: `binary_logloss` / `auc`.
  * Early stopping: 50 rounds.
  * Generate Out-Of-Fold (OOF) predicted probabilities for every candidate pair across all 5 folds.
  * Save trained models or ensemble fold models for test inference.

---

### STEP 9: OOF Decision Gate & Metric Maximization
* **Target File:** `code/business_entity_resolution/src/decision.py`
* **Responsibility:** Optimize entity-level decision thresholds on OOF predictions against macro-$F_{0.5}$.
* **Decision Rule:**
  1. For each S1 entity:
     * Let $P_{\max} = \max_{j} P_j$.
     * If $P_{\max} < \tau_{\text{singleton}}$, predict $\emptyset$ (protect singleton score).
     * Else, emit candidate $j$ if:
       $$P_j \ge \tau_{\text{match}} \quad \text{and} \quad (P_{\max} - P_j) \le \Delta_{\text{prob}}$$
  2. Perform grid search on OOF predictions over:
     * $\tau_{\text{singleton}} \in [0.4, 0.8]$ (step 0.05)
     * $\tau_{\text{match}} \in [0.4, 0.8]$ (step 0.05)
     * $\Delta_{\text{prob}} \in [0.1, 0.3]$ (step 0.05)
  3. Store best $(\tau_{\text{singleton}}^*, \tau_{\text{match}}^*, \Delta_{\text{prob}}^*)$ and print final benchmark OOF macro-$F_{0.5}$.

---

### STEP 10: End-to-End Test Inference & Submission Verification
* **Target File:** `code/business_entity_resolution/src/main.py`
* **Responsibility:** Execute end-to-end pipeline on test set and run official validator.
* **Execution Steps:**
  1. Load `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`.
  2. Normalize test records.
  3. Run candidate blocking $\rightarrow$ write `output/candidate_pairs.tsv`.
  4. Extract features for test pairs.
  5. Score with trained ensemble model.
  6. Apply optimized decision gate $\rightarrow$ write `output/matching_results.tsv`.
  7. Run official submission validator:
     ```powershell
     python "d:/programming/amazon ml challange/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py" `
       --matching "d:/programming/amazon ml challange/amazon-ml-hackathon-2026-submission/output/matching_results.tsv" `
       --candidate "d:/programming/amazon ml challange/amazon-ml-hackathon-2026-submission/output/candidate_pairs.tsv" `
       --test-dir "d:/programming/amazon ml challange/6ab10eb3b23ba_student_resource/student_resource/dataset/test"
     ```
  8. Assert validator output prints `PASS` with exit code 0.

---

## 🚦 Execution Checklist for the Agent
| Step | Module | Command to Run | Expected Outcome |
| :---: | :--- | :--- | :--- |
| **1** | Dependencies | `pip install lightgbm rapidfuzz scikit-learn` | No import errors |
| **2** | Config | `python -m src.config` | Paths resolved correctly |
| **3** | Metrics | `python -m src.metrics` | Unit tests PASS (returns 0.714 and 1.0) |
| **4** | Blocking | `python -m src.blocking --sample 10000` | Candidate recall printed ($\ge 90\%$) |
| **5** | Features | `python -m src.features --sample 1000` | Feature matrix generated with zero NaN |
| **6** | Training | `python -m src.model --sample 10000` | 5-fold CV trained, OOF predictions saved |
| **7** | Decision | `python -m src.decision` | Best $\tau$ found, OOF $F_{0.5}$ score logged |
| **8** | Main | `python -m src.main` | Valid `matching_results.tsv` and `candidate_pairs.tsv` |
| **9** | Validator | `python utils/validate_submission.py ...` | `PASS (exit 0)` |
