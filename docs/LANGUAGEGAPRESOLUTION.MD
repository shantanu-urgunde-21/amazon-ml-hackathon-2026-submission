# Cross-Lingual & Cross-Script Resolution Rules
**Amazon ML Challenge 2026 — Business Entity Resolution**

---

## 1. Indic Scripts (Devanagari, Bengali, Tamil, Telugu, Kannada, Gujarati, Malayalam, Odia, Gurmukhi)
* **What the data contains:** S1 is always Latin script. S2/S3 India records put English words into Indic script (`राम मार्केटिंग प्राइवेट लिमिटेड` = `Ram Marketing Private Limited`).
* **Why rules fail:** `unidecode` gives `raam maarkettiNg praaivett limittedd`. The earlier "schwa correction" gave `भारत` → `bhart` and `प्राइवेट` → `praivet`, never the English spelling.
* **Fix in `normalization/transliterate.py`:** a lexicon learned from fit-split training links, aligning Indic words with their S1 English words (see `DATASET_LANGUAGE_AND_SCRIPT_ANALYSIS.md` §3). `anyascii` is the fallback for words the lexicon does not know.

---

## 2. Phonetic Consonant Skeletons
To ensure variant spellings collide in the same blocking bucket:
* **Canonicalize Equivalents:** `ph` $\to$ `f`, `w` $\to$ `v`, `sh` $\to$ `s`, `z` $\to$ `j`, `ck`/`c`/`q` $\to$ `k`, `th` $\to$ `t`, `dh` $\to$ `d`, `bh` $\to$ `b`.
* **Vowel Stripping:** Retain first character; strip interior vowels (`a, e, i, o, u, y, h`).
* **Consonant Deduplication:** Collapse repeated consonants (`ll` $\to$ `l`, `ss` $\to$ `s`).
* **Examples** (`normalization/text.py: phonetic_key`, as implemented):
  * `shri` / `sri` / `shree` $\to$ `sr`
  * `balaji` $\to$ `blj`
  * `enterprises` $\to$ `entrprs`

---

## 3. French Language Adaptations (Unseen Test Country)
* **Ligature Expansion:** Map `œ` $\to$ `oe`, `æ` $\to$ `ae`, `ß` $\to$ `ss` before NFKD normalization.
* **Elision Splitting:** Replace apostrophes (`'`, `’`) and hyphens (`-`) with spaces (`l'oréal` $\to$ `l oreal`, `rue de l'église` $\to$ `rue de l eglise`).
* **Prefix Thoroughfares:** Street designators appear at the start in France (`Rue de la Paix`, `Avenue Victor Hugo`, `Boulevard Haussmann`).
* **Address Modifiers:** Capture sub-modifiers (`14 bis`, `12 ter`, `14B`) alongside base house number (`14`).
* **CEDEX / BP Filtering:** Commercial postal designations (`CEDEX 01`, `BP 123`) are stripped so they do not conflict with postal codes.

