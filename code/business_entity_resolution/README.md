# Business Entity Resolution Pipeline
**Amazon ML Challenge 2026**  
**Task:** Match Source 1 business entities to all corresponding Source 2 and Source 3 records.

---

## 🏗️ Architecture Overview

The system is built as a modular two-stage retrieval-and-ranking pipeline optimized for macro-averaged $F_{0.5}$:

```
Raw Sources (S1, S2, S3)
        │
        ▼
1. Multilingual Normalization & Structured Slot Extraction
   ├── NFKD Unicode & French ligature expansion (œ, æ, ß)
   ├── Schwa-corrected Devanagari transliteration via unidecode
   ├── Phonetic consonant skeletons (e.g., shree/sri -> sr, balaji -> blj)
   └── Address slot extraction (PIN/ZIP, building/plot number, unit)
        │
        ▼
2. Scalable Inverted Index Blocking
   ├── Rare name token index (frequency bucket capped)
   ├── Cross-script phonetic skeleton index
   ├── Numeric address token index
   └── Character 3-gram index
        │
        ▼
3. Pairwise Feature Engineering
   ├── C++ string similarities (Levenshtein, Jaro-Winkler, Token-Sort/Set via rapidfuzz)
   ├── Address & numeric token overlap Jaccard
   ├── Ternary contradiction states (+1 agree, 0 missing, -1 conflict)
   └── Pre-GBDT candidate-relative context (margins, candidate rank, bucket size)
        │
        ▼
4. Gradient Boosted Classifier (LightGBM)
   └── 5-Fold GroupKFold strictly split by source1_entity_id
        │
        ▼
5. Entity-Level Decision Gate
   └── Threshold search (tau_singleton, tau_match, delta_prob) maximizing Macro-F0.5
        │
        ▼
Output Submission TSVs
   ├── output/candidate_pairs.tsv
   └── output/matching_results.tsv
```

---

## 📁 Source Package Structure

All implementation code is cleanly partitioned into dedicated modules under `src/`:

```
code/business_entity_resolution/src/
├── config.py                 # Central path resolution and hyperparameters
├── main.py                   # Master end-to-end pipeline runner
├── dataloader/               # High-throughput streaming TSV loaders
│   ├── __init__.py
│   └── loader.py
├── normalization/            # Multilingual cleaning, transliteration & slots
│   ├── __init__.py
│   └── normalizer.py
├── filters/                  # Inverted index blocking & recall evaluation
│   ├── __init__.py
│   └── blocking.py
├── features/                 # C++ pairwise similarity & contradiction features
│   ├── __init__.py
│   └── extractor.py
├── models/                   # GroupKFold LightGBM binary classifier
│   ├── __init__.py
│   └── classifier.py
├── prediction/               # Macro-F0.5 decision gate & threshold optimizer
│   ├── __init__.py
│   └── gate.py
├── validation/               # Exact competition macro-F0.5 metric & splitters
│   ├── __init__.py
│   ├── metrics.py
│   └── splitters.py
└── notebooks/                # Exploratory notebooks
    └── README.md
```

---

## 🚀 How to Run & Reproduce

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Fast Verification (Sample Mode)
To verify the end-to-end pipeline on 2,000 entities in ~10 seconds:
```bash
python src/main.py --sample 2000
```

### 3. Run Full Pipeline (Training + Test Inference)
To train the full model and generate official submission files:
```bash
python src/main.py
```

### 4. Validate Submission Format
```bash
python ../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test
```