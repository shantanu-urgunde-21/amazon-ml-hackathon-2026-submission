# Dataset Language & Script Distribution Analysis
**Amazon ML Challenge 2026 — Business Entity Resolution**

---

## 1. Empirical Audit Findings

Audit across 100,000-record batches per file revealed a **cross-script asymmetry**:
* **Source 1 (Reference):** **0.00% non-Latin**. All records (including India) are in the Latin alphabet.
* **Source 2 & 3 (Targets):** **~41% of Indian records** are in native Indic scripts (24% Devanagari, 17% other Indic).
* **Impact:** Standard ASCII fuzzy matching yields `0.0` between English S1 and Devanagari S2/S3 without transliteration.

| File | Country Distribution | Devanagari (% of India) | Other Indic Scripts | Non-Latin Total |
| :--- | :--- | :---: | :---: | :---: |
| **`train_source1.tsv`** | India: 40.1%, US: 59.9% | **0.00%** | **0.00%** | **0.00%** |
| **`train_source2.tsv`** | India: 40.1%, US: 59.9% | **23.52%** | **16.97%** | **16.22%** |
| **`train_source3.tsv`** | India: 40.5%, US: 59.5% | **18.77%** | **13.17%** | **12.93%** |
| **`test_source1.tsv`**  | India: 46.6%, US+FR: 53.4% | **0.00%** | **0.00%** | **0.00%** |
| **`test_source2.tsv`**  | India: 47.3%, US+FR: 52.7% | **24.01%** | **17.39%** | **19.60%** |
| **`test_source3.tsv`**  | India: 47.3%, US+FR: 52.7% | **18.86%** | **13.82%** | **15.46%** |

*(Note: Test set introduces ~13% France records; India expands to ~47%).*

---

## 2. Key Failure Cases

1. **Full Devanagari Name:**
   * S1: `Ram Marketing Private Limited` $\leftrightarrow$ S2: `राम मार्केटिंग प्राइवेट लिमिटेड`
   * *ASCII similarity = 0.0.* Requires phonetic transliteration to intersect.
2. **Code-Mixed Names:**
   * S1: `Sun Power Provision` $\leftrightarrow$ S3: `Sun पावर Provision`
3. **Appended State/City in Address:**
   * `PLOT NO B-78/1... महाराष्ट्र` (Maharashtra) or `... मध्य प्रदेश` (Madhya Pradesh).

---

## 3. Implemented Solutions

1. **Language-Agnostic Numeric Anchors:** PIN codes and plot numbers (`400093`, `B-78/1`) are Arabic numerals across all scripts; indexed with frequency caps in `blocking.py`.
2. **Schwa-Corrected Transliteration & Skeletons:** `unidecode` + schwa stripping (`kamala` $\to$ `kamal`) + phonetic skeleton reduction (`shree`/`sri` $\to$ `sr`, `balaji` $\to$ `blj`) in `normalizer.py`.
3. **Cross-Script GBDT Indicators:** `is_cross_script` boolean feature in `extractor.py` allows the classifier to rely on phonetic similarity and numeric agreement when scripts diverge.

