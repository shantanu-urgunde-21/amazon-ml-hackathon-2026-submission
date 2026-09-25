# Experiment & Benchmark Tracker
**Amazon ML Challenge 2026 — Business Entity Resolution**  
**Last Run:** 2026-09-25 22:23:24  

---

## 📈 Experiment Progression Summary

| Run ID | Experiment Tag | Features | Recall | Macro F0.5 | Non-Sing F0.5 | Singleton Acc | India F0.5 | US F0.5 | Runtime | Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `01` | **Phase-2-Structured** | 23 | 86.31% | **0.9413**  | 0.9250 | 97.9% | N/A | N/A | 45.88s | Phase 2: Multilingual Transliteration, Address Slots, Ternary Contradiction, and Group Margins |
| `02` | **Phase-2-Structured** | 23 | 86.30% | **0.9413** (-0.0000) | 0.9239 | 98.2% | N/A | N/A | 44.02s | Phase 2: Multilingual Transliteration, Address Slots, Ternary Contradiction, and Group Margins |
| `03` | **Phase-2-Structured** | 23 | 86.29% | **0.9415** (+0.0002) | 0.9262 | 97.7% | 0.9094 | 0.9622 | 45.85s | Phase 2: Multilingual Transliteration, Address Slots, Ternary Contradiction, and Group Margins |

---

## 🔬 Failure Attribution & Error Audit (Run `03`: Phase-2-Structured)
**Description:** Phase 2: Multilingual Transliteration, Address Slots, Ternary Contradiction, and Group Margins  

### 1. Error Attribution Breakdown
- **Stage 1 — Blocking Misses:** `1,755` true links dropped during candidate blocking (upper recall bound).
- **Stage 2 — GBDT False Negatives:** `241` true links were retrieved by blocking, but scored below decision threshold.
- **Stage 3 — False Merges (Non-Singletons):** `62` candidate pairs falsely merged with an S1 entity.
- **Stage 4 — Singleton False Positives:** `35` singleton entities were falsely linked (hurting 1.0 credit).

### 2. Country-Specific Sub-Scores
- **India Macro-F0.5:** `0.9094`
- **US Macro-F0.5:** `0.9622`

### 3. Concrete Row-Level Error Inspections
#### A. Sample Blocking Misses (Lost at Candidate Retrieval)
- **S1:** `S1-890009542` | **Name:** SJ Ace Vendome Inc | **Addr:** 60 Uneeda Street, Madison, WV
  **True Target:** `S2-461344955` | **Name:** sjacevendome.com | **Addr:** 60 UNEEDA STREET, MADISON, WV
  *Reason: Never retrieved in candidate blocking*

- **S1:** `S1-890009542` | **Name:** SJ Ace Vendome Inc | **Addr:** 60 Uneeda Street, Madison, WV
  **True Target:** `S2-932187797` | **Name:** #sjace | **Addr:** UNEEDA STREET, MADISON, WV
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
  **Falsely Predicted:** `S3-230826619` (P=0.964) | **Name:** Sindt and Swaney Corp | **Addr:** 36870 Van Brocklin Rd, Crathage CITY, New York

- **S1:** `S1-321648485` | **Name:** Herman Total Software LLC | **Addr:** 1151 Birdwell Drive, Gallatin, TN
  **Falsely Predicted:** `S2-375083807` (P=0.936) | **Name:** Herman Total Restaurant [Llc] | **Addr:** 1164 BIRDWELL DR, GALLATIN, TN

- **S1:** `S1-988131510` | **Name:** Hunger Guild | **Addr:** Eagle Mountain, UT, 1132 Hoof Print Drive
  **Falsely Predicted:** `S2-355055649` (P=0.990) | **Name:** Hunger Guild Holdings | **Addr:** 1133 HOOF PRINT DRIVE, EAGLE MOUNTAIN, UT

- **S1:** `S1-908758233` | **Name:** Congregation Church | **Addr:** 929 Ridge Road, Lewiston, NY
  **Falsely Predicted:** `S3-673223808` (P=0.897) | **Name:** Congregation Báptist Church Co | **Addr:** 

- **S1:** `S1-672951415` | **Name:** Hyderabad Research Limited | **Addr:** Flat No.202, No.8/2/456, Road No.4, Banjara Hills, Hyderabad, Telangana
  **Falsely Predicted:** `S2-203449639` (P=0.935) | **Name:** Hyderabad Research Limited | **Addr:** PLOT NO: D 20/PART, PHASE-I, IDA, JEEDIMETLA, HYDERABAD, Telangana

#### C. Sample Singleton False Merges (Singletons Lost Credit)
- **S1 (True Singleton):** `S1-141278611` | **Name:** Compass LLC | **Addr:** 820 Center Street, Unit 2, Provo, UT
  **Falsely Predicted:** `S3-656229637` (P=0.962) | **Name:** Compass LLC | **Addr:** 521-A Center Street, # 20, Ludlow, Massachusetts

- **S1 (True Singleton):** `S1-141278611` | **Name:** Compass LLC | **Addr:** 820 Center Street, Unit 2, Provo, UT
  **Falsely Predicted:** `S3-914897360` (P=0.997) | **Name:** Compass Llc | **Addr:** 521-A Center Street, Unit 20, Ludlow, Massachusetts

- **S1 (True Singleton):** `S1-338296041` | **Name:** Suryaksh Buildtech Private Limited | **Addr:** 16-11-51, Moosarambagh Malakpet, Hyderabad, Telangana
  **Falsely Predicted:** `S2-985441801` (P=0.681) | **Name:** వైట్ సాఫ్ట్‌వేర్ ప్రైవేట్ లిమిటెడ్ | **Addr:** 16-11-781/39, MOOSARAMBAGH, AMBERPET, HYDERABAD, Telangana

- **S1 (True Singleton):** `S1-670289577` | **Name:** Pediatric Dental Physicians Inc | **Addr:** 632 Twin Brooks Drive, Mount Vernon, WA
  **Falsely Predicted:** `S3-538882911` (P=0.986) | **Name:** Pediatric Dental Physicians Inc. | **Addr:** 

- **S1 (True Singleton):** `S1-784822153` | **Name:** Regional Foundation | **Addr:** 652 Front Street, Mounds, IL
  **Falsely Predicted:** `S2-347177489` (P=0.890) | **Name:** REGIONAL  FOUNDATION III | **Addr:** 216 ASH STREET, COBEN, IL

---
### 4. Top Feature Importances (GBDT Gain)
| Rank | Feature Name | Importance Gain |
| :---: | :--- | :---: |
| 1 | `addr_token_sort` | 179535.4 |
| 2 | `addr_token_set` | 174727.2 |
| 3 | `name_jaro_winkler` | 12602.8 |
| 4 | `name_token_sort` | 9156.4 |
| 5 | `addr_levenshtein` | 5296.5 |
| 6 | `addr_numeric_jaccard` | 4796.6 |
| 7 | `phonetic_jaccard` | 3774.4 |
| 8 | `building_number_state` | 3689.1 |

### 5. Latency Breakdown
- **Normalization:** `4.14s`
- **Candidate Blocking:** `4.0s`
- **Feature Extraction:** `6.46s`
- **Model Training (5 Folds):** `9.04s`
- **Total Time:** `45.85s`

---