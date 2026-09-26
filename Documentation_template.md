# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Antigravity ER Team
**Team Members:** [fill in]
**Submission Date:** September 26, 2026

---

## 1. Executive Summary

We normalize multilingual business records, retrieve candidates with FAISS similarity indexes, and score them with XGBoost. Indic-script text is translated with a lexicon learned from the training links. Candidates come from **reverse assignment**: every S2/S3 record looks up its nearest S1 records by name and by address in exact FAISS indexes partitioned by country and state. This keeps the candidate set to **10.5 candidates per S1 entity** on the test set (8.4 on the holdout), while keeping **94.5% of true links** on held-out training entities. On an untouched holdout of training entities the pipeline scores a **macro F0.5 of 0.9537** (India 0.9418, US 0.9617).

---

## 2. Methodology

### 2.1 Problem Analysis

* **Scale:** 1.7M test S1 entities against 10.0M S2/S3 records. Train: 2.2M against 10.3M.
* **Structure of the links** (train ground truth): every S2/S3 record matches **at most one** S1 entity, and links never cross countries. An S1 entity has 3.46 matches on average (median 3, 99th percentile 8), and 5.6% of entities have none. 26% of pool records match nothing (distractors).
* **Scripts:** S1 is always Latin script. In S2/S3, 41% of India records contain Indic script in the name or address: Devanagari 24%, plus Bengali, Tamil, Telugu, Kannada, Gujarati, Malayalam, Odia and Gurmukhi. These are mostly *English words written in Indic script* (`प्राइवेट लिमिटेड` = private limited), so rule-based transliteration fails.
* **Name noise:** case, accents, OCR digits (`Appare1s`), legal forms anywhere (`LLC Espinoza…`, `L.L.C.`, `Pvt.`), honorifics (`Sri`, `Dr`), junk prefixes (`--`, `***`), brackets, word reordering, typos, aliases (`X DBA Y`, `X formerly Y`, `X née Y`), domains (`lifeinvestments.com`, `Name | www.x.com`, `CORALIEFETESSASCOM`), and appended phone numbers.
* **Address noise:** reordered components; states as code, name or native script; abbreviations (`St`/`Street`/`Saint`, `Rd`, French `R.`/`Bd`/`All.`); `City of X`, `X CDP`; house-number formats (`004512`, `#1998`, `C-2-08` vs `C-208`, `506-510`); PO box / PMB / CEDEX; `null` tokens; truncation; missing addresses (~3%). Postcodes are practically absent.
* **France** exists only in the test set (15% of test S1). Its legal forms (SARL, SAS, EURL, SCI, SNC…), street types and region/department names are handled by rules. No labels are used for it.

### 2.2 Solution Strategy

**Approach type:** normalization → FAISS similarity-index blocking → gradient-boosted pair classifier → entity-level decision gate.
**Core innovation:** reverse-assignment retrieval. Because each S2/S3 record belongs to at most one S1 entity, the pool records query an index of S1 records, not the other way round. Each S1 entity's candidate set is then only the records that point at it. This gives small, adaptive candidate sets instead of a fixed top-K.

---

## 3. Candidate Generation (Blocking)

* **Vectors:** char 2-4-gram (word-bounded) TF-IDF of the normalized name and of the normalized address, compressed with truncated SVD to 128 dimensions each (`retrieval/embedder.py`). Fitted unsupervised on each country's pool.
* **Indexes:** exact FAISS flat inner-product indexes (GPU `GpuIndexFlatIP` when available, else CPU `IndexFlatIP`; identical results) over S1 records, one per **country × state partition** and per view (name, address). True matches agree on state in > 98.8% of links; WA/DC and AP/Telangana share a partition because the data mixes them. Records without a state search all partitions of their country.
* **Keys per pool record:** its best S1 by name alone, its best by address alone, its best by the combined similarity (0.4 · name + 0.6 · address), plus up to 3 more within 0.05 of that best. Records without an address keep up to 5 by name within 0.10.
* **Candidate pairs generated (test):** 18,169,981 (10.49 per S1 entity; France 11.47, India 10.92, US 9.58; cap 30). That is 0.0003% of all same-country pairs (reduction ratio 0.999997).
* **How true matches are protected:** three retrieval views (name / address / combined), the state partitions fall back to country-wide search, and the one-to-one structure means a record's true S1 is almost always its nearest S1. Holdout candidate recall: **94.46%** at 8.4 candidates per entity.

---

## 4. Matching Model

**Features (36, `features/extractor.py`):**

* **Name:** Levenshtein ratio, token-set, token-sort and partial ratio, Jaro-Winkler, exact match, no-space ratio and partial ratio (for domains), best alias similarity (DBA/formerly), phonetic-key token set, legal-form agreement (+1/0/-1), token counts, first-token equality.
* **Address:** token-set, ratio and partial ratio, empty-address flags, state agreement (+1/0/-1), house-number agreement (+1/0/-1), number-code token set.
* **Retrieval:** combined / name / address cosine, forward rank within the S1 entity's candidates, reverse rank of the S1 among the record's nearest S1s, margin to the record's best S1.
* **Candidate-set context:** size, rank and margin of the pair within its S1 entity by cosine and by name similarity; source (S2/S3) and Indic flag.

**Model type:** XGBoost binary classifier (Apache-2.0; `hist` trees, trained on the GPU when available), 5-fold GroupKFold by S1 entity, trained on the candidate pairs of 300k fit-split entities. The fold models are averaged at inference.

**Post-processing:**

1. **Exclusive assignment:** a pool record is linked to at most one S1 entity, the most probable one.
2. **Decision gate:** if max P < τ_singleton, predict no match. Otherwise take the candidates with P ≥ τ_match and max P − P ≤ Δ_prob.

**Threshold selection:** grid search maximizing macro F0.5 on one half of the holdout entities (τ_singleton = 0.65, τ_match = 0.05, Δ_prob = 0.30).

---

## 5. Results & Error Analysis

Validation protocol: train S1 entities are split by entity into fit (80%) and holdout (20%). The lexicon, model and retrieval settings use fit only. The gate is tuned on half the holdout, and the score is reported on the other half, which nothing touched (details in the code README).

| Metric (holdout report half, 221,025 entities) | Value |
| :--- | ---: |
| **Macro F0.5** | **0.9537** |
| Macro F0.5 India / US | 0.9418 / 0.9617 |
| Macro F0.5 on the tuning half | 0.9540 |
| Candidate recall | 94.46% |
| Candidates per S1 entity | 8.4 |
| Pair precision / pair recall | 98.78% / 90.38% |
| Singletons correctly left empty | 94.5% |

True links lost: 41,669 at blocking and 31,901 by the model or gate, out of 764,624. France has no labels. Its predictions (3.49 matches per entity, 5.0% empty) look like the training distribution (3.46 and 5.6%).

* **Common false positives (wrong merges):** 8,520 wrong pairs out of 699,574 predicted. 57% link a record that belongs to no S1 entity, and 43% link a record that belongs to another entity. Typical cases: the same name at a neighbouring house number (`306` vs `310 Monroe St`, same unit); a different business at the same address (`Chathams Trusted Textiles` vs `Zleigrex`, both `501 Rebecca Smith Way`); and name-only records without an address whose name also fits another entity.
* **Common false negatives (missed matches):** (1) blocking misses: short generic names with truncated addresses in large cities (mostly India), and records with no address; (2) model rejections: same name but a noisy or different house number (`10` vs `8 Sunflower Dr`), heavily garbled names (`endocsnoolryg`), and addressless records whose name is shared by several entities.

---

## 6. Conclusion

Treating the one-record-one-entity structure as a retrieval constraint gives small candidate sets (8–11 per entity, with adaptive size) without giving up recall. A lexicon learned from the training links turns Indic-script records into comparable English text. The main remaining gaps are addressless records and generic names in dense cities. Better handling of those, for example with a learned embedding, would raise both the recall ceiling (94.5%) and the model recall.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (entry point `src/main.py`; stages in `src/pipeline.py`; see its `README.md`):

```bash
make init
cp code/business_entity_resolution/config.example.toml code/business_entity_resolution/config.toml
make run          # writes output/candidate_pairs.tsv and output/matching_results.tsv
```

### B. Additional Results

| Model (same holdout half) | Macro F0.5 | India | US | Train time |
| :--- | ---: | ---: | ---: | ---: |
| LightGBM, CPU, 1,000 rounds, coarse gate grid | 0.9525 | 0.9400 | 0.9609 | 8.5 min |
| **XGBoost, GPU, early stopping, finer gate grid** | **0.9537** | **0.9418** | **0.9617** | **5.6 min** |

| Retrieval (fit entities) | Candidate recall | Candidates per S1 |
| :--- | ---: | ---: |
| Combined vector top-1, one global flat index (US) | 91.8% | 4.7 |
| + name-only / address-only views, state partitions (US) | 95.6% | 7.8 |
| + wider set for addressless records, noise fixes (US) | 95.8% | 8.1 |
| same, India | 92.4% | 8.8 |

Most important features (XGBoost total gain): address number-code overlap, address token-set similarity, reverse-assignment margin, house-number agreement, legal-form agreement, name partial ratio, name cosine.
