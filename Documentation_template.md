# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Antigravity ER Team  
**Task:** Business Entity Resolution (Source 1 $\rightarrow$ Source 2 & Source 3)  
**Submission Date:** September 25, 2026  

---

## 1. Executive Summary

We present a high-throughput, multilingual retrieval-and-ranking pipeline specifically engineered for the competition's **Macro-averaged $F_{0.5}$** metric. The pipeline resolves reference entities from Source 1 against noisy candidate targets in Source 2 and Source 3 across US, Indian, and unseen French business records. 

Key technical innovations include:
1. **Multilingual Script Normalization:** Inherent-schwa-corrected transliteration (`kamala` $\to$ `kamal`) and phonetic consonant skeletons (`shree`/`sri` $\to$ `sr`, `balaji` $\to$ `blj`) bridging Latin S1 and native Devanagari S2/S3 records.
2. **Two-Tier Compound Inverted Index Blocking:** Scalable 5-way blocking indexing rare tokens, compound keys (`token#p_{pin}`, `token#s_{state}`) for frequent corporate words, phonetic skeletons, numeric tokens, and multi-token 3-grams.
3. **Structured Ternary Contradiction Logic (+1 / 0 / -1):** Physical address slot contradictions (street number conflicts on identical streets, administrative state conflicts, PIN mismatches) engineered to aggressively defend precision under the $F_{0.5}$ penalty.
4. **Candidate-Relative Group Context:** Group-level feature engineering (top candidate margins, candidate rank, and bucket volume) feeding a 5-fold `GroupKFold` LightGBM classifier paired with an entity-level decision gate ($\tau_{\text{singleton}}, \tau_{\text{match}}, \Delta_{\text{prob}}$).

On our 5,000-entity benchmark, the system achieves an Out-Of-Fold **Macro-$F_{0.5}$ of 0.9421** (India: 0.9097, US: 0.9629) with **98.1% singleton accuracy** and end-to-end execution in under 45 seconds.

---

## 2. Methodology

### 2.1 Problem Analysis
* **Cross-Script Asymmetry:** Empirical analysis revealed that Source 1 is **100% Latin script**, whereas Source 2 and 3 contain **~41% native Indic scripts** (Devanagari, Bengali, Tamil, Telugu) in Indian records. Standard ASCII fuzzy matching yields `0.0` similarity, permanently dropping true matches during blocking unless transliterated.
* **Open-Set Country Shift:** The test set introduces unseen records from **France** (~13%), requiring robust handling of French elisions (`l'`, `d'`), ligatures (`œ`, `æ`), prefix street designators (`Rue de la Paix`), and administrative boilerplate (`CEDEX`).
* **$F_{0.5}$ Metric Economics:** Precision is penalized $2\times$ over recall ($1/\beta^2 = 4$). Singletons earn **1.0** for an empty prediction, but drop to **0.0** upon a single false merge. Error audits proved that defending precision against false merges delivers ~3× higher score ROI than hunting low-frequency minority scripts.

### 2.2 Solution Strategy
* **Approach Type:** Two-Stage Retrieval (Inverted Index Blocking) + High-Dimensional Ranking (LightGBM) + Precision-Defensive Decision Gating.
* **Core Innovation:** Language-agnostic numeric anchors combined with phonetic consonant skeleton projections and ternary contradiction slots (+1 agreement, 0 missing, -1 explicit contradiction).

---

## 3. Candidate Generation (Blocking)

To reduce the $O(N \times M)$ pairwise comparison space down to $\le 50$ high-probability candidates per S1 entity without recall leakage:

- **Blocking Keys Used:**
  1. **Rare Name Tokens:** Informative tokens appearing in $\le 1,500$ records with corporate stop-words suppressed.
  2. **Compound Frequent Tokens:** For high-frequency non-generic corporate names (*Precision*, *Continental*, *Universal*, *Healthcare* appearing in $>1,500$ records), compound keys are formed: `token#p_{postal_code}` and `token#s_{state}`.
  3. **Phonetic Consonant Skeletons:** Consonant projections mapped via canonical phonetic equivalences (`ph` $\to$ `f`, `w` $\to$ `v`, `sh` $\to$ `s`, `ck`/`c`/`q` $\to$ `k`, interior vowels stripped).
  4. **Address Numeric Anchors:** Extracted PIN codes, building numbers, and plot identifiers.
  5. **Informative 3-Grams:** Character 3-grams across the first two non-trivial name tokens for typo and sub-string tolerance.
- **Candidate Volume & Coverage:** Generates an average of **49.4 candidates per S1 entity**, achieving **86.5% candidate recall** on the training ground truth.

---

## 4. Matching Model

### Features Used (28 Total Features)
1. **Name Similarities (7):** Levenshtein ratio, token-sort ratio, token-set ratio, Jaro-Winkler, exact match boolean, length difference ratio, prefix match boolean (via C++ accelerated `rapidfuzz`).
2. **Address Similarities (4):** Token-set ratio, token-sort ratio, Levenshtein ratio, exact address match boolean.
3. **Numeric Token Overlap (2):** Numeric token intersection count and numeric Jaccard overlap.
4. **Metadata & Cross-Script (3):** Country equality indicator, candidate source indicator (`cand_is_s2`), cross-script indicator (`is_cross_script`), and phonetic skeleton Jaccard similarity.
5. **Ternary Contradiction Slots (5):**
   - `postal_code_state`: $+1$ (equal), $0$ (missing), $-1$ (conflict)
   - `building_number_state`: $+1$ (equal), $0$ (missing), $-1$ (conflict)
   - `unit_state`: $+1$ (equal), $0$ (missing), $-1$ (conflict)
   - `street_num_conflict_same_street`: $1.0$ if street names match ($\ge 0.80$) but primary street numbers conflict
   - `state_region_state` / `geo_conflict`: $+1$ (same state), $0$ (missing), $-1$ (state conflict)
6. **Candidate-Relative Group Context (3):**
   - `cand_rank_name_sim`: Candidate rank within the S1 retrieval pool
   - `cand_margin_name_sim`: Distance in name similarity from the top candidate in the group
   - `cand_bucket_size`: Total candidate pool size for S1

### Model & Decision Gate
- **Classifier:** 5-Fold `GroupKFold` LightGBM binary classifier grouped strictly by `source1_entity_id` to eliminate data leakage.
- **Decision Gate:** Post-GBDT entity-level optimization searching over $(\tau_{\text{singleton}}, \tau_{\text{match}}, \Delta_{\text{prob}})$ directly maximizing Macro-$F_{0.5}$. If $\max_j P_j < \tau_{\text{singleton}}$, the entity is predicted as an empty set to protect the 1.0 singleton credit; otherwise, candidate $j$ is emitted if $P_j \ge \tau_{\text{match}}$ and $(\max_j P_j - P_j) \le \Delta_{\text{prob}}$.

---

## 5. Results & Error Analysis

| Metric | Baseline (v0.1) | Phase 2 (Slots & Margins) | Phase 3 (Contradictions & Compound) |
| :--- | :---: | :---: | :---: |
| **Macro $F_{0.5}$** | 0.9399 | 0.9415 | **0.9421** |
| **Non-Singleton $F_{0.5}$** | 0.9221 | 0.9262 | **0.9255** |
| **Singleton Accuracy** | 97.4% | 97.7% | **98.1%** |
| **India Macro $F_{0.5}$** | 0.8980 | 0.9094 | **0.9097** |
| **US Macro $F_{0.5}$** | 0.9602 | 0.9622 | **0.9629** |
| **Blocking Recall** | 86.10% | 86.29% | **86.48%** |

### Error Post-Mortems
- **Common False Positives (False Merges):**
  1. *Adjacent House Numbers:* High name and street similarity causing merges across neighboring plots (e.g. `36861` vs `36870` Van Brocklin Rd). Mitigated via `street_num_conflict_same_street`.
  2. *Regional Multi-Branches:* Common corporate names (*Compass LLC*, *Hyderabad Research Limited*) across different cities/states. Mitigated via positional state extraction and `state_region_state`.
- **Common False Negatives (Missed Matches):**
  1. *Un-spaced Domain Names:* S1 business names matching URL targets (e.g. `sjacevendome.com`). Mitigated via domain token unpacking in `normalizer.py`.
  2. *Infrequent Transliterations:* Minority Dravidian scripts (Telugu, Tamil) in Indian records without Latin overlap.

---

## 6. Conclusion

By treating business entity resolution as a precision-first problem under Macro-$F_{0.5}$, our architecture pairs linguistically grounded normalization and compound blocking with structured physical contradiction logic. The pipeline processes 5,000 entities in 44 seconds and scales linearly across the 1.7M test set without external APIs.

---

## Appendix

### A. Code Artefacts & Structure
The complete implementation is self-contained in `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── requirements.txt           # Pinned dependencies (lightgbm, rapidfuzz, unidecode, scikit-learn)
├── README.md                  # Complete reproduction walkthrough
└── src/
    ├── config.py              # Central path resolution and hyperparameters
    ├── main.py                # End-to-end pipeline CLI orchestrator
    ├── dataloader/loader.py   # Streaming high-throughput TSV loaders
    ├── normalization/normalizer.py # Multilingual cleaning, slots & transliteration
    ├── filters/blocking.py    # 5-way inverted index with compound keys
    ├── features/extractor.py  # 28 pairwise string, contradiction & group features
    ├── models/classifier.py   # 5-fold GroupKFold LightGBM classifier
    ├── prediction/gate.py     # OOF decision gate maximizing Macro-F0.5
    └── validation/metrics.py  # Official Macro-F0.5 evaluation metric
```

**Entry Point to Reproduce Outputs:**
```bash
cd code/business_entity_resolution
pip install -r requirements.txt
python src/main.py
```
Outputs are written strictly to:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

### B. Validation Verification
The generated submission files pass all checks in `validate_submission.py`:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
