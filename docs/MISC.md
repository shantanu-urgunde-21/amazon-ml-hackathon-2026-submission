# Miscellaneous Findings, Error Audit & Technical Backlog
**Amazon ML Challenge 2026 — Business Entity Resolution**

---

## 1. Documentation Map
* `TASK_CHECKLIST.md`: Milestone progress and status tracker.
* `AGENT_EXECUTION_PLAN.md`: System architecture and module contracts.
* `EXPERIMENT_TRACKER.md`: Benchmark history, CV metrics, and failure attribution.
* `DATASET_LANGUAGE_AND_SCRIPT_ANALYSIS.md`: Empirical script distribution and cross-script strategy.
* `LANGUAGEGAPRESOLUTION.MD`: Linguistic edge cases (Devanagari schwa, French elisions).
* `MISC.md`: Error post-mortems, $F_{0.5}$ metric economics, and technical backlog.

---

## 2. $F_{0.5}$ Metric Economics & ROI
$$F_{0.5} = \frac{1.25 \times \text{TP}}{1.25 \times \text{TP} + 0.25 \times \text{FP} + 1.0 \times \text{FN}}$$

* **Precision is weighted $2\times$ over Recall ($1 / \beta^2 = 4$).**
* **Singleton Penalty:** Singletons earn **1.0** if empty. A single false merge drops the entity score to **0.0**.
* **Benchmark Error Distribution (5,000 S1 sample):**
  * Stage 1 (Blocking Misses): **1,755** true links dropped (recall ceiling = 86.29%).
  * Stage 2 (GBDT Score Drops): **241** true links scored below threshold.
  * Stages 3 & 4 (False Merges): **97** pairs (62 non-singleton + 35 singleton).
* **ROI Takeaway:** Eradicating the 97 false merges yields ~3× faster $F_{0.5}$ score gains than hunting edge-case minority scripts.

---

## 3. Failure Post-Mortems & Solutions

### A. Street Number Contradiction (Precision Leak)
* **Cases:**
  * `Sindt and Swaney Inc` at `36861 Van Brocklin Rd` falsely merged with `36870 Van Brocklin Rd` ($P=0.964$).
  * `Hunger Guild` at `1132 Hoof Print Dr` falsely merged with `1133 Hoof Print Dr` ($P=0.990$).
* **Cause:** High name (~95%) and street name similarity overwhelms numeric mismatch.
* **Fix:** Extract primary street number; apply ternary feature: $+1$ (match), $0$ (missing), $-1$ (conflict on matching street).

### B. Common Name Disambiguation Across Locations
* **Cases:**
  * `Compass LLC` in Provo, UT merged with `Compass LLC` in Ludlow, MA ($P=0.962$).
  * `Hyderabad Research Limited` in Banjara Hills merged with same name in Jeedimetla ($P=0.935$).
* **Cause:** Generic corporate words (*Compass*, *Research*, *Holdings*) yield 1.0 name similarity.
* **Fix:** Token IDF weighting (`name_specificity = max(IDF(tokens))`) + geographic hard contradiction (state/city conflict triggers $P=0$).

### C. Blocking Recall Ceiling (Bucket Caps)
* **Problem:** 1,755 true links missed due to `MAX_TOKEN_BUCKET_SIZE = 1500` dropping frequent words (*Precision*, *Continental*, *Universal*).
* **Fix:** Two-tier compound indexing: index frequent tokens paired with postal code (`token_pin`) or state (`token_state`).

### D. Un-spaced Domain Names
* **Case:** `SJ Ace Vendome Inc` missed target `sjacevendome.com` (0 shared tokens).
* **Fix:** Strip `.com`, `.in`, `www.` and split concatenated tokens in `normalizer.py`.

### E. One-Sided Missing Address
* **Case:** `Pediatric Dental Physicians Inc` (WA) falsely merged with empty-address target ($P=0.986$).
* **Fix:** Add `address_missing_either` feature; require higher prediction threshold when address corroboration is absent.

---

## 4. Prioritized Engineering Backlog

| ID | Priority | Description | Target | Expected Gain |
| :---: | :---: | :--- | :---: | :---: |
| `B-01` | **P1** | Primary street number conflict penalty ($-1$ state) | Precision | **+0.015** |
| `B-02` | **P1** | State/city contradiction & token IDF weighting | Precision | **+0.012** |
| `B-03` | **P2** | Compound inverted index blocking (`token_pin`) | Recall | **+0.015** |
| `B-04` | **P2** | Domain / URL splitting (`.com`, `.in`, `www.`) | Recall | **+0.005** |
| `B-05` | **P3** | Empty-address confidence gate for singletons | Precision | **+0.004** |
| `B-06` | **P4** | Telugu & Tamil script transliteration | Recall | **+0.002** |
| `B-07` | **P5** | Dense multilingual embeddings (FAISS) | Recall | Experimental |

