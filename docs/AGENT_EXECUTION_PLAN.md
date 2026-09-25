# Pipeline Architecture & Operational Playbook
**Amazon ML Challenge 2026 — Business Entity Resolution**

---

## 1. Engineering Guardrails
1. **No Un-indexed Quadratic Loops:** Training & test data exceed 2.5 GB (~1.7M test entities). Candidate generation must pass through indexed blocking.
2. **Format Strictness:** Always use `sep="\t"` with header matching official submission requirements.
3. **Competition Rules:** Zero external APIs/lookups allowed (automatic disqualification).
4. **Metric Integrity:** Macro-$F_{0.5}$ weights precision $2\times$ over recall. Singletons receive full 1.0 credit if left empty, and 0.0 upon any false merge.

---

## 2. Modular Architecture & Data Flow

```
Raw TSVs (S1, S2, S3)
       │
       ▼
1. dataloader/loader.py: Streaming TSV & ground truth mapping dictionaries
       │
       ▼
2. normalization/normalizer.py: NFKD, ligature expansion, schwa-corrected transliteration, address slots
       │
       ▼
3. filters/blocking.py: Scalable inverted index (rare tokens, phonetic skeletons, numbers, 3-grams)
       │
       ▼
4. features/extractor.py: RapidFuzz C++ similarities, address slot states (+1/0/-1), group margin context
       │
       ▼
5. models/classifier.py: 5-Fold GroupKFold LightGBM binary classifier
       │
       ▼
6. prediction/gate.py: Threshold grid search (tau_singleton, tau_match, delta_prob) maximizing Macro-F0.5
       │
       ▼
7. Output Submission TSVs: output/candidate_pairs.tsv & output/matching_results.tsv
```

---

## 3. Core Module Contracts

| Module | Core Responsibility | Key Interfaces |
| :--- | :--- | :--- |
| `src/config.py` | Central paths & hyperparameters | `TRAIN_S1_PATH`, `TEST_S1_PATH`, `MAX_TOKEN_BUCKET_SIZE` |
| `src/dataloader/loader.py` | Streaming I/O & ground truth loading | `load_tsv()`, `load_ground_truth()` |
| `src/normalization/normalizer.py` | Multilingual cleaning & slot parsing | `normalize_record()`, `phonetic_skeleton()` |
| `src/filters/blocking.py` | Inverted index candidate generation | `InvertedIndexBlocker`, `evaluate_recall()` |
| `src/features/extractor.py` | C++ fuzzy similarity & slot contradictions | `FeatureExtractor.extract_batch()` |
| `src/models/classifier.py` | Grouped LightGBM model training & OOF scoring | `EntityResolutionModel.train_cv()` |
| `src/prediction/gate.py` | Precision-defense entity decision gate | `DecisionGate.tune_thresholds()`, `predict()` |
| `src/validation/metrics.py` | Exact competition macro-$F_{0.5}$ metric | `macro_f05()`, `compute_s1_fbeta()` |

---

## 4. Verification & Validation Commands

```bash
# Fast verification (sample 2,000 entities in ~10s)
python src/main.py --sample 2000

# Full dataset training & test inference
python src/main.py

# Official submission validator
python ../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
