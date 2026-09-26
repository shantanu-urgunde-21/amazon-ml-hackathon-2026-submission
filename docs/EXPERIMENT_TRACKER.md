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
| `08` | **FAISS-SVD-Vector-XGBoost-GPU** | 22 | **94.45%** | **0.9482** | — | — | **0.9350** | **0.9614** | 8.0 min (train) | Full 2.2M dataset: 128d SVD FAISS retrieval (8.35 cands/S1), GPU XGBoost on 300k entities, 221k holdout benchmark |
| `09` | **Fix-1-OCR-Overlap-CountryGates** | 26 | **94.45%** | **0.9525** (+0.0043) | — | — | **0.9426** | **0.9623** | 8.7 min (train) | Full 221k holdout benchmark: Added OCR ratio, unobserved address feature, addr_nums_overlap (#8 top feature), and country-specific gating (India 0.9426, US 0.9623). |
| `10` | **Phase-0-Bidirectional-Gate-Widening** | 26 | 94.45% | **0.9542** (tune) / **0.9509** (rep) | — | — | **0.9426** | **0.9623** | 2.5 min | Widened boundary truncation: `tau_match` [0.05..0.40] -> [0.01..0.60], `delta_prob` [0.30..1.00] -> [0.05..1.00]. Proved both India and US naturally optimize at `tau_match = 0.55` (was artificially clamped at 0.40). |
| `11` | **Phase-1-Granular-Miss-Diagnostic** | 26 | 94.45% | — | — | — | — | — | 3.1 min | Vectorized audit of 4,882 holdout blocking misses. Disproved partition hypothesis: wrong-partition was only 0.1% (India: 2/2,689) and 0.9% (US: 20/2,193). Falsified & skipped Phase 2 per gate rule (<15%). Proved tight reverse-margin (`rev_k=3, rev_margin=0.05`) prematurely cut true matches retrieved in top-17. Relaxed to `rev_k=6, rev_margin=0.10` (+2.0% recall). |
| `12` | **Phase-3-Metric-Projector-PyTorch-CUDA** | 26 | **95.43%** (+3.15%) | *In Progress* | — | — | — | — | ~12 min | Supervised MetricResidualProjector (128->256->128 MLP) trained on fit split with InfoNCE loss + hard negatives. True match cosine 10th percentile jumped 0.6472 -> 0.8127. Holdout blocking recall surged 92.28% -> 95.43% while candidate pool dropped from 2.49 to 2.31 per entity. Integrated TorchGpuIndexFlatIP on NVIDIA GTX 1650 for 3.5x faster search with dynamic query batching (<200 MB VRAM). |


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

## 🔬 Deep-Dive Diagnostic & Supervised Metric Learning (Runs 10–12)

### 1. Partition Miss Diagnostic Results & Falsification of Phase 2
- **Audit Methodology:** Evaluated all 4,882 blocking misses on 25k holdout entities across India and US (`scratch/inspect_misses_detailed.py`).
- **Hypothesis Tested:** Did hard state partitioning cause the majority of blocking misses?
- **Empirical Finding:**
  - India: Wrong-partition misses accounted for only **0.1%** (2 out of 2,689 misses).
  - US: Wrong-partition misses accounted for only **0.9%** (20 out of 2,193 misses).
  - **Verdict:** The state partitioning hypothesis was overwhelmingly **falsified** (<1% vs the >15% threshold required to justify cross-partition expansion). Phase 2 (soft partitioning) was skipped, avoiding massive candidate explosion.
- **Downstream Keep-Filter Discovery:**
  - True matches (such as `sweet nails`) were frequently retrieved by FAISS inside the raw top-17 candidates, but subsequently dropped by overly aggressive reverse-margin pruning (`rev_k=3, rev_margin=0.05`).
  - Widening reverse-margin filters to `rev_k=6, rev_margin=0.10` captured +2.0% recall directly without modifying embeddings.

### 2. Supervised Metric Projector Architecture & Validation
- **Architecture (`src/retrieval/projector.py`):**
  - Residual MLP ($128 \to 256 \to 128$) with LayerNorm and GELU activations.
  - Initialized with zero-weight final linear layer to strictly preserve baseline identity mapping at step 0.
  - Trained via contrastive InfoNCE loss ($\tau=0.07$) with hard false candidate mining mined from fit-split negatives.
- **Leakage Boundary:**
  - Projectors for Name and Address are fit strictly on the 80% `fit` split of train S1 entities. Zero holdout information is seen.
- **Validation Lift:**
  - True match cosine distribution 10th percentile on untouched holdout jumped from **0.6472 to 0.8127 (+0.1655)**.
  - Blocking candidate recall evaluated on 15,000 holdout links surged from **92.28% to 95.43% (+3.15% net lift)**.
  - Candidate density improved simultaneously: average candidates per S1 dropped from **2.49 to 2.31**, demonstrating higher discriminative precision.

### 3. GPU Hardware Acceleration (`TorchGpuIndexFlatIP`)
- **Bottleneck Identified:**
  - On Windows, `faiss-gpu` is not available on PyPI, causing FAISS to fall back to single-threaded CPU search across all partition indexes. Candidate generation on 6.18M US pool records took ~45 minutes on CPU.
- **Implementation (`src/retrieval/index.py`):**
  - Implemented `TorchGpuIndexFlatIP` utilizing PyTorch CUDA on the local **NVIDIA GeForce GTX 1650 (4 GB VRAM)**.
  - Employs dynamic query batching (`chunk_q` bounded to keep intermediate matrix multiplications strictly $<200\text{ MB}$ VRAM).
  - Search latency dropped from ~4 minutes per 500k chunk to **~1 minute 11 seconds (3.5x speedup)**.
  - Total train/US candidate generation (6.18M pool records) completed in **17 minutes** with zero VRAM leaks.

### 4. Known Shortcomings & Bottlenecks
1. **Ambiguous Name-Only Entities & Acronym Mismatches:**
   - Entities lacking address information and possessing extreme typographical abbreviations or acronyms (e.g., `#sjace` vs `SJ Ace Vendome Inc`) remain challenging for character 3-gram embeddings alone.
2. **Macro $F_{0.5}$ Precision Sensitivity:**
   - Under $F_{0.5}$ ($\beta=0.5$), false positive links are penalized $2\times$ as severely as false negative misses. Arbitrarily relaxing retrieval parameters ($k > 6$) inflates false candidate volume, which degrades downstream decision-gate precision and singleton identification.
3. **Hardware Memory Envelope (4 GB VRAM):**
   - The 4 GB VRAM ceiling necessitates batched inner-product multiplications and explicit memory garbage collection (`torch.cuda.empty_cache()`) between country partitions.