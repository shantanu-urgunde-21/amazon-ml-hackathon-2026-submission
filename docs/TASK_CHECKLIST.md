# Task Tracker & Milestone Checklist
**Amazon ML Challenge 2026 — Business Entity Resolution**  
**Current Milestone:** Phase 3 Complete (CV Macro-$F_{0.5} = 0.9421$) | **Overall Progress:** ~90%

---

## 📊 Phase Progress Summary

| Phase | Description | Status | Completion |
| :--- | :--- | :---: | :---: |
| **Phase 0** | Setup, Environment & Dataset Placement | 🟢 Done | 100% |
| **Phase 1** | v0.1 Fast End-to-End Baseline Pipeline | 🟢 Done | 100% |
| **Phase 2** | High-Yield Enhancements (Slots, Contradictions, Margins) | 🟢 Done | 100% |
| **Phase 3** | Precision & Recall Fixes (Backlog `B-01` to `B-04`) | 🟢 Done | 100% |
| **Phase 4** | Multilingual Dense Retrieval (FAISS / BGE-M3) | ⚪ Optional | 0% |
| **Phase 5** | Test Inference, Submission Verification & Packaging | 🟡 Active | 75% |

---

## 📌 Phase Breakdown

### Phase 0: Setup & Diagnostics (100% Complete)
- [x] Directory structure and path resolution in `src/config.py`.
- [x] Pinned dependencies in `requirements.txt` (`lightgbm`, `rapidfuzz`, `scikit-learn`, `unidecode`).
- [x] Streaming TSV loader verified on 2.2M ground truth pairs.

### Phase 1: v0.1 Baseline (100% Complete)
- [x] Exact macro-$F_{0.5}$ metric engine with singleton credit (1.0) and false merge penalty (0.0).
- [x] Inverted index blocking with bucket caps (`MAX_TOKEN_BUCKET_SIZE=1500`) and 3-grams.
- [x] Baseline C++ string & numeric overlap features (`rapidfuzz`).
- [x] 5-Fold `GroupKFold` LightGBM binary classifier.
- [x] OOF entity-level decision gate ($\tau_{\text{singleton}}, \tau_{\text{match}}, \Delta_{\text{prob}}$).
- [x] Output generator verified against `validate_submission.py`.

### Phase 2: High-Yield Enhancements (100% Complete)
- [x] Schwa-corrected Devanagari transliteration + phonetic consonant skeletons (`sr`, `blj`).
- [x] Structured address slot extraction (`postal_code`, `building_number`, `unit_slot`).
- [x] Ternary contradiction features (+1 agreement, 0 missing, -1 conflict).
- [x] Group-relative context features (`cand_rank_name_sim`, `cand_margin_name_sim`, `cand_bucket_size`).
- [x] Benchmark validation: Macro-$F_{0.5}$ reached **0.9415** (India: 0.9094, US: 0.9622).

### Phase 3: Targeted Error Fixes (100% Complete)
- [x] **`B-01` Primary Street Number Contradiction:** Extracted `primary_street_num` and `street_name`; added `street_num_state`, `street_name_sim`, and `street_num_conflict_same_street`.
- [x] **`B-02` State/City Contradiction:** Full state dictionary + positional 2-letter regex; added `state_region_state` and `geo_conflict`.
- [x] **`B-03` Compound Blocking:** Implemented `token#p_{pin}` and `token#s_{state}` for frequent corporate tokens; upgraded 3-gram indexing across multi-word tokens.
- [x] **`B-04` Domain/URL Tokenizer:** Cleaned URLs (`.com`, `.in`, `www.`) and stripped hashtags/mentions.
- [x] Benchmark validation: Macro-$F_{0.5}$ improved to **0.9421**; singleton accuracy reached **98.1%**.

### Phase 4: Optional Semantic Retrieval (On Hold)
- [ ] Test compact multilingual embedding model (e.g. `bge-m3` $\le 560\text{M}$) if lexical blocking ceiling stays below 95%.

### Phase 5: Submission & Verification (75% Complete)
- [x] Verified submission output structure and format compatibility.
- [x] Filled official methodology document: `Documentation_template.md`.
- [x] Clean reproduction documentation in `code/business_entity_resolution/README.md`.
- [ ] Run full test set inference on `dataset/test/` (including unseen France records).
- [ ] Validate final `output/candidate_pairs.tsv` and `output/matching_results.tsv` via `validate_submission.py`.
- [ ] Build final submission zip via `make package`.


