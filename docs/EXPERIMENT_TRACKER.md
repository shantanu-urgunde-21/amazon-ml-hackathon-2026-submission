# Experiment & Benchmark Tracker
**Amazon ML Challenge 2026 — Business Entity Resolution**  
**Last Run:** 2026-09-26 11:26:46  

> **Superseded (2026-09-26).** Runs `01`–`06` below were produced with `main.py --sample 5000`
> and a pipeline version that is no longer in the repository. They cannot be reproduced. The current
> code's `--sample` mode also took the first N rows of every file independently, so almost no true
> matches were in the sample. The reproducible benchmark is the holdout evaluation in
> `cache/benchmark.json` (see `code/business_entity_resolution/README.md` → Results).

---

## 📈 Experiment Progression Summary

| Run ID | Experiment Tag | Features | Recall | Macro F0.5 | Non-Sing F0.5 | Singleton Acc | India F0.5 | US F0.5 | Runtime | Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `01` | **Phase-2-Structured** | 23 | 86.31% | **0.9413**  | 0.9250 | 97.9% | N/A | N/A | 45.88s | Phase 2: Multilingual Transliteration, Address Slots, Ternary Contradiction, and Group Margins |
| `02` | **Phase-2-Structured** | 23 | 86.30% | **0.9413** (-0.0000) | 0.9239 | 98.2% | N/A | N/A | 44.02s | Phase 2: Multilingual Transliteration, Address Slots, Ternary Contradiction, and Group Margins |
| `03` | **Phase-2-Structured** | 23 | 86.29% | **0.9415** (+0.0002) | 0.9262 | 97.7% | 0.9094 | 0.9622 | 45.85s | Phase 2: Multilingual Transliteration, Address Slots, Ternary Contradiction, and Group Margins |
| `04` | **Phase-3-Contradictions** | 28 | 86.54% | **0.9426** (+0.0010) | 0.9259 | 98.1% | 0.9103 | 0.9633 | 47.24s | Phase 3: Street Number Contradiction, State Conflict, and Compound Blocking |
| `05` | **Phase-3-Precision-Defense** | 28 | 86.50% | **0.8980** (-0.0445) | 0.8623 | 98.1% | 0.8977 | 0.8982 | 48.76s | Phase 3: Contradiction Hard Penalty Gating + Compound Blocking + URL Cleaner |
| `06` | **Phase-3-Clean-Contradictions** | 28 | 86.48% | **0.9421** (+0.0441) | 0.9255 | 98.1% | 0.9097 | 0.9629 | 44.48s | Phase 3: Clean State & Street Contradictions + Compound Blocking + Domain Cleaner |
| `07` | **Phase-3-10k-Random-Sample** | 28 | 87.15% | **0.9369** | 0.9347 | 97.2% | 0.8965 | 0.9636 | 92.46s | Phase 3: 10000 S1 Random Evaluation with Full Pool Ground Truth Alignment |

---

## 🔬 Failure Attribution & Error Audit (Run `06`: Phase-3-Clean-Contradictions)
**Description:** Phase 3: Clean State & Street Contradictions + Compound Blocking + Domain Cleaner  

### 1. Error Attribution Breakdown
- **Stage 1 — Blocking Misses:** `1,730` true links dropped during candidate blocking (upper recall bound).
- **Stage 2 — GBDT False Negatives:** `259` true links were retrieved by blocking, but scored below decision threshold.
- **Stage 3 — False Merges (Non-Singletons):** `57` candidate pairs falsely merged with an S1 entity.
- **Stage 4 — Singleton False Positives:** `31` singleton entities were falsely linked (hurting 1.0 credit).

### 2. Country-Specific Sub-Scores
- **India Macro-F0.5:** `0.9097`
- **US Macro-F0.5:** `0.9629`

### 3. Concrete Row-Level Error Inspections
#### A. Sample Blocking Misses (Lost at Candidate Retrieval)
- **S1:** `S1-890009542` | **Name:** SJ Ace Vendome Inc | **Addr:** 60 Uneeda Street, Madison, WV
  **True Target:** `S2-932187797` | **Name:** #sjace | **Addr:** UNEEDA STREET, MADISON, WV
  *Reason: Never retrieved in candidate blocking*

- **S1:** `S1-890009542` | **Name:** SJ Ace Vendome Inc | **Addr:** 60 Uneeda Street, Madison, WV
  **True Target:** `S2-461344955` | **Name:** sjacevendome.com | **Addr:** 60 UNEEDA STREET, MADISON, WV
  *Reason: Never retrieved in candidate blocking*

- **S1:** `S1-832050262` | **Name:** Shakti Agro Limited | **Addr:** C/O Gurnav Singh Saluja, Beside Zudio, Kultapara, Sadar, Sambalpur, Orissa
  **True Target:** `S3-673592452` | **Name:** Wexveo | **Addr:** Block F-264- C/o Gurnav Singh Saluja, Beside Zudio, Kultapara, Sadar, Sambalpur, OD
  *Reason: Never retrieved in candidate blocking*

- **S1:** `S1-874660306` | **Name:** Physical Therapy Associates Group | **Addr:** 4303 Elkins Avenue, Unit B, Nashville, TN
  **True Target:** `S2-279857860` | **Name:** Group Physical Therapy Asseatiaes | **Addr:** 004303 ELKINS AVE, PO BOX 5980, NASHVILLE, TN
  *Reason: Never retrieved in candidate blocking*

- **S1:** `S1-109593962` | **Name:** Great Impex Private Limited | **Addr:** 6-2-101/5/C, Telangana, Hyderabad, Secunderabad, Lane Beside Centralview Apt New Bhoiguda
  **True Target:** `S3-118565228` | **Name:** గ్రేట్ ఇంపెక్స్ ప్రైవేట్ లిమిటెడ్ | **Addr:** 6-2-101/5/C, Lane Beside Centralview Apt New Bhoiguda, Secunderabad, Hyderabad, TG
  *Reason: Never retrieved in candidate blocking*

#### B. Sample False Merges (Precision Errors — Model Falsely Predicted Match)
- **S1:** `S1-383740360` | **Name:** Sindt and Swaney Inc | **Addr:** Champion, 36861 Van Brocklin Road, NY
  **Falsely Predicted:** `S3-230826619` (P=0.989) | **Name:** Sindt and Swaney Corp | **Addr:** 36870 Van Brocklin Rd, Crathage CITY, New York

- **S1:** `S1-321648485` | **Name:** Herman Total Software LLC | **Addr:** 1151 Birdwell Drive, Gallatin, TN
  **Falsely Predicted:** `S2-375083807` (P=0.929) | **Name:** Herman Total Restaurant [Llc] | **Addr:** 1164 BIRDWELL DR, GALLATIN, TN

- **S1:** `S1-988131510` | **Name:** Hunger Guild | **Addr:** Eagle Mountain, UT, 1132 Hoof Print Drive
  **Falsely Predicted:** `S2-355055649` (P=0.993) | **Name:** Hunger Guild Holdings | **Addr:** 1133 HOOF PRINT DRIVE, EAGLE MOUNTAIN, UT

- **S1:** `S1-958475533` | **Name:** Big Auto Body | **Addr:** 201 High Ridge Trail, Rio Rancho, NM
  **Falsely Predicted:** `S2-926656275` (P=0.906) | **Name:** BIG ÁUTO ÁUTO BÓDY | **Addr:** 

- **S1:** `S1-672951415` | **Name:** Hyderabad Research Limited | **Addr:** Flat No.202, No.8/2/456, Road No.4, Banjara Hills, Hyderabad, Telangana
  **Falsely Predicted:** `S2-203449639` (P=0.961) | **Name:** Hyderabad Research Limited | **Addr:** PLOT NO: D 20/PART, PHASE-I, IDA, JEEDIMETLA, HYDERABAD, Telangana

#### C. Sample Singleton False Merges (Singletons Lost Credit)
- **S1 (True Singleton):** `S1-141278611` | **Name:** Compass LLC | **Addr:** 820 Center Street, Unit 2, Provo, UT
  **Falsely Predicted:** `S3-656229637` (P=0.971) | **Name:** Compass LLC | **Addr:** 521-A Center Street, # 20, Ludlow, Massachusetts

- **S1 (True Singleton):** `S1-141278611` | **Name:** Compass LLC | **Addr:** 820 Center Street, Unit 2, Provo, UT
  **Falsely Predicted:** `S3-914897360` (P=0.998) | **Name:** Compass Llc | **Addr:** 521-A Center Street, Unit 20, Ludlow, Massachusetts

- **S1 (True Singleton):** `S1-338296041` | **Name:** Suryaksh Buildtech Private Limited | **Addr:** 16-11-51, Moosarambagh Malakpet, Hyderabad, Telangana
  **Falsely Predicted:** `S2-985441801` (P=0.840) | **Name:** వైట్ సాఫ్ట్‌వేర్ ప్రైవేట్ లిమిటెడ్ | **Addr:** 16-11-781/39, MOOSARAMBAGH, AMBERPET, HYDERABAD, Telangana

- **S1 (True Singleton):** `S1-670289577` | **Name:** Pediatric Dental Physicians Inc | **Addr:** 632 Twin Brooks Drive, Mount Vernon, WA
  **Falsely Predicted:** `S3-538882911` (P=0.979) | **Name:** Pediatric Dental Physicians Inc. | **Addr:** 

- **S1 (True Singleton):** `S1-784822153` | **Name:** Regional Foundation | **Addr:** 652 Front Street, Mounds, IL
  **Falsely Predicted:** `S2-347177489` (P=0.726) | **Name:** REGIONAL  FOUNDATION III | **Addr:** 216 ASH STREET, COBEN, IL

---
### 4. Top Feature Importances (GBDT Gain)
| Rank | Feature Name | Importance Gain |
| :---: | :--- | :---: |
| 1 | `addr_token_set` | 282713.7 |
| 2 | `addr_token_sort` | 45329.9 |
| 3 | `addr_levenshtein` | 27584.1 |
| 4 | `name_jaro_winkler` | 13615.0 |
| 5 | `name_token_sort` | 8534.2 |
| 6 | `building_number_state` | 7100.2 |
| 7 | `phonetic_jaccard` | 4370.8 |
| 8 | `state_region_state` | 2736.8 |

### 5. Latency Breakdown
- **Normalization:** `5.42s`
- **Candidate Blocking:** `4.1s`
- **Feature Extraction:** `6.32s`
- **Model Training (5 Folds):** `6.72s`
- **Total Time:** `44.48s`

---