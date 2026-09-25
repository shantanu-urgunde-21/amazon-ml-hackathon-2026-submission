# Cross-Lingual & Cross-Script Resolution Rules
**Amazon ML Challenge 2026 — Business Entity Resolution**

---

## 1. Devanagari (Hindi/Marathi) Transliteration
* **The Inherent-Schwa Problem:** In Devanagari, consonants have an implicit `'a'` vowel unless silenced by a virama (`्`). Naive `unidecode` transliterates `"भारत"` $\to$ `"bhaarata"` (instead of `"bharat"`) and `"कमल"` $\to$ `"kamala"` (instead of `"kamal"`).
* **Fix in `normalizer.py`:**
  * Transliterate via `unidecode`.
  * Collapse doubled vowels (`aa` $\to$ `a`, `ee`/`ii` $\to$ `i`, `oo`/`uu` $\to$ `u`).
  * Collapse doubled retroflexes (`tt` $\to$ `t`, `dd` $\to$ `d`).
  * Strip trailing schwa `'a'` on words $>3$ characters ending in a consonant + `'a'`.

---

## 2. Phonetic Consonant Skeletons
To ensure variant spellings collide in the same blocking bucket:
* **Canonicalize Equivalents:** `ph` $\to$ `f`, `w` $\to$ `v`, `sh` $\to$ `s`, `z` $\to$ `j`, `ck`/`c`/`q` $\to$ `k`, `th` $\to$ `t`, `dh` $\to$ `d`, `bh` $\to$ `b`.
* **Vowel Stripping:** Retain first character; strip interior vowels (`a, e, i, o, u, y, h`).
* **Consonant Deduplication:** Collapse repeated consonants (`ll` $\to$ `l`, `ss` $\to$ `s`).
* **Examples:**
  * `"श्री"` / `"shri"` / `"sri"` $\to$ `sr`
  * `"बालाजी"` / `"balaji"` $\to$ `blj`
  * `"एंटरप्राइजेज"` / `"enterprises"` $\to$ `entrprjs`

---

## 3. French Language Adaptations (Unseen Test Country)
* **Ligature Expansion:** Map `œ` $\to$ `oe`, `æ` $\to$ `ae`, `ß` $\to$ `ss` before NFKD normalization.
* **Elision Splitting:** Replace apostrophes (`'`, `’`) and hyphens (`-`) with spaces (`l'oréal` $\to$ `l oreal`, `rue de l'église` $\to$ `rue de l eglise`).
* **Prefix Thoroughfares:** Street designators appear at the start in France (`Rue de la Paix`, `Avenue Victor Hugo`, `Boulevard Haussmann`).
* **Address Modifiers:** Capture sub-modifiers (`14 bis`, `12 ter`, `14B`) alongside base house number (`14`).
* **CEDEX / BP Filtering:** Commercial postal designations (`CEDEX 01`, `BP 123`) are stripped so they do not conflict with postal codes.

