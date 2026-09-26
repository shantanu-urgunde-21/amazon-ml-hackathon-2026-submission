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

The table above was re-checked on 2026-09-26 and holds. It counts records with Indic text in the name *or* the address. Counting names only, 23% of India records in S2 and 13% in S3 have an Indic name.

1. **Learned Indic → English lexicon** (`normalization/transliterate.py`). Indic records are mostly English words written in Indic script, so rule-based romanization cannot recover them (`कंस्ट्रक्शन` → `kmstrksn`). The lexicon is learned from training links of the **fit split only**:
   * names: an Indic name and its S1 name with the same token count are aligned word by word (95% of Indic↔English name pairs have equal length; mapping purity 97%);
   * addresses: Indic components are aligned with S1 components, keeping mappings that co-occur in ≥ 80% of pairs. This recovers state names, including one-word-to-two-word cases (`தமிழ்நாடு` → `tamil nadu`, `পশ্চিমবঙ্গ` → `west bengal`).
   * About 1,330 Indic words are learned. They cover 84% of the distinct Indic words in the test set. The rest fall back to `anyascii` romanization.
2. **Phonetic keys** (`normalization/text.py: phonetic_key`) bridge the rough fallback romanization and typos (`shree`/`sri`/`shri` → `sr`, `balaji` → `blj`, `motors` / `मोटर्स` → `mtrs`). They are used as a model feature.
3. **Char n-gram retrieval + numbers**. Candidates come from char 2-4-gram vectors of normalized names and addresses (FAISS), so a partially translated or romanized record still shares most n-grams with its S1 record. House numbers and PIN-like digits are identical across scripts.
