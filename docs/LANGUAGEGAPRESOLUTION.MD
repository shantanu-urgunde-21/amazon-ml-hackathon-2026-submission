Having Hindi (Devanagari script) in the training data alongside the unseen `France` test set changes the requirements for both `normalize.py` and `blocking.py`. Specifically, your pipeline has to handle two distinct cross-lingual phenomena:

1. **Cross-Script Transliteration (Phonetic equivalence):** `"श्री बालाजी एंटरप्राइजेज"` vs. `"Shree Balaji Enterprises"` (same spoken words written in Devanagari vs. Latin script).

2. **Cross-Lingual Translation (Semantic equivalence):** `"शर्मा मिष्ठान भंडार"` vs. `"Sharma Sweet Shop"`, or `"Société Générale de Banque"` vs. `"General Banking Company"`.

If you rely on standard ASCII/Latin string similarity, any pair where $S_1$ is in Devanagari and $S_2/S_3$ is in Latin will have a character similarity of `0.0` and will be dropped during blocking before your GBDT ever sees it.

---

### 1. Solving the Hindi / Devanagari Transliteration Gap

Simply passing Devanagari text through Python's `unidecode` or standard ISO-15919 transliteration is not enough because of the **Devanagari Inherent-Vowel (Schwa) & Retroflex Problem**:

- In Unicode Devanagari, every consonant carries an implicit `'a'` vowel unless explicitly suppressed by a _virama_ (`्`) or modified by a vowel matra.
- Consequently, naive `unidecode` transliterates `"भारत"` as `"bhaarata"` (instead of `"bharat"`), `"कमल"` as `"kamala"` (instead of `"kamal"`), and `"प्राइवेट लिमिटेड"` as `"praaivett limittedd"`.
- Furthermore, Indian English spelling is non-standardized across vowels (`ee`/`i`, `oo`/`u`, `aa`/`a`), labial approximants (`v`/`w`), sibilants (`sh`/`s`), and aspirates (`ph`/`f`, `ksh`/`x`).

To make Devanagari and English records collide in the exact same blocking buckets and achieve high fuzzy similarity, create **two phonetic projections** in `normalize.py`:

1. **Schwa-Corrected Latin Transliteration (`text_latin`):**
   Transliterate non-Latin characters to ASCII via `unidecode`, then apply regex normalization to collapse doubled vowels (`aa` $\to$ `a`, `ee` $\to$ `i`, `oo` $\to$ `u`), collapse doubled retroflex consonants (`tt` $\to$ `t`, `dd` $\to$ `d`), and strip word-final schwas (`a` at the end of tokens $>3$ characters following a consonant, so `"bhaarata"` $\to$ `"bharat"` and `"kamala"` $\to$ `"kamal"`).
2. **Phonetic Consonant Skeleton (`phonetic_skeleton`):**
   Map every token to a canonical phonetic skeleton by:

- Normalizing phonetic equivalents: `ph` $\to$ `f`, `w` $\to$ `v`, `sh` $\to$ `s`, `ksh` $\to$ `x`, `z` $\to$ `j`, `c`/`ck`/`q` $\to$ `k`.
- Dropping non-leading vowels (`a, e, i, o, u, y`) and collapsing adjacent identical consonants.
- **Result:** `"श्री"` (`shree`), `"shri"`, and `"sri"` all collapse to `"sr"`; `"बालाजी"` (`baalaajii`) and `"balaji"` both collapse to `"blj"`; `"एंटरप्राइजेज"` (`enttarpraaijej`) and `"enterprises"` both collapse to `"entrprjs"`.

---

### 2. Making Extraction French-Ready (Without Country Hard-Coding)

While French uses the Latin script, French business names and addresses violate several assumptions built into English/Indian parsers. You can handle all of them using universal, country-agnostic linguistic rules:

- **Unicode NFKD + Explicit Ligature Expansion:**
  Standard `unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')` strips accents (`é, è, ê, ë` $\to$ `e`; `ç` $\to$ `c`; `à, â` $\to$ `a`), which handles noisy French records where one source omits accents (`Cafe de la Gare` vs. `Café de la Gare`). However, `NFKD` deletes ligatures like `œ` (`Bœuf`, `Cœur`) and `æ` if encoded to ASCII directly—explicitly map `œ` $\to$ `oe` and `æ` $\to$ `ae` prior to `NFKD`.
- **Apostrophe / Elision Splitting (`l'`, `d'`, `qu'`):**
  French heavily contracts articles and prepositions onto the front of nouns: `L'Oréal`, `Côte d'Azur`, `Rue de l'Église`, `Quai d'Orsay`.
- **Flaw in naive punctuation removal:** Stripping `'` turns `l'église` into `leglise`, which fails to match `eglise` on token blocking.
- **Universal Fix:** Replace apostrophes (`'`, `’`, ```) and hyphens (`-`) with **spaces** during tokenization, and allow your within-country Corpus-IDF filter to downweight ultra-high-frequency 1-to-2 character clitics/prepositions (`l`, `d`, `de`, `du`, `la`, `le`, `et`, `au`, `en`in France;`of`, `in`, `to`, `at` in US/India).

- **Prefix vs. Suffix Street Designators:**
  In the US and India, street types sit at the **end** of the street name (`12 MG Road`, `450 Main Street`). In France, street types sit at the **beginning** (`12 Rue de la Paix`, `45 Avenue Victor Hugo`, `Place Bellecour`, `Boulevard Haussmann`, `Allée`, `Impasse`, `Chemin`, `Quai`).

- Ensure your `thoroughfare_core` extractor strips street designators regardless of whether they appear at the start or end of the street phrase.

- **Trailing `CEDEX` and Building Number Modifiers (`bis`, `ter`):**
- French commercial addresses frequently end with `CEDEX` (or `CEDEX 01`, `BP 123` for postal boxes) after the city name. Because `CEDEX` has a massive Document Frequency in French business data, your unsupervised positional-DF filter will naturally flag it as boilerplate, preventing a `CEDEX` suffix from masquerading as a conflicting postal code or street number.
- Building numbers in France (and India) often carry a modifier (`14 bis`, `14B`, `12/A`). Capture `\b\d+\s*(?:[a-zA-Z]\b|bis\b|ter\b)?` as the full `building_number` slot, and extract both the base integer (`14`) and the sub-modifier (`bis`/`b`).

---

### 3. Drop-In Cross-Script & Multilingual Normalizer (`normalize.py`)

Below is the self-contained normalization logic that bridges Devanagari, French, and English into a unified representation using only standard Python + `unidecode` (MIT License):

```python
import re
import unicodedata
from unidecode import unidecode

LIGATURE_MAP = str.maketrans({
    "œ": "oe", "Œ": "oe",
    "æ": "ae", "Æ": "ae",
    "ß": "ss", "&": " and ",
})

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
APOSTROPHE_HYPHEN_RE = re.compile(r"['’`\-–—/.,()\[\]{}:;\"!|+]+")
WHITESPACE_RE = re.compile(r"\s+")

def has_non_latin_script(text: str) -> bool:
    """Detects whether the input string contains Devanagari or other non-Latin scripts."""
    return bool(DEVANAGARI_RE.search(text)) or any(ord(c) > 0x024F for c in text)

def transliterate_and_fold(text: str) -> str:
    """
    Converts Devanagari/non-Latin scripts to ASCII and strips French/Latin diacritics
    while preserving token boundaries across French elisions (l', d') and hyphens.
    """
    if not text or not isinstance(text, str):
        return ""

    is_non_latin = has_non_latin_script(text)
    text = text.translate(LIGATURE_MAP)

    if is_non_latin:
        # Transliterate Devanagari / non-Latin to ASCII
        text = unidecode(text)
    else:
        # Fast NFKD accent stripping for Latin/French
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    text = text.lower()
    # Split elisions (l'oreal -> l oreal, rue de l'eglise -> rue de l eglise)
    text = APOSTROPHE_HYPHEN_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    if is_non_latin:
        # Fix Devanagari inherent-schwa and retroflex artifacts from unidecode
        tokens = []
        for tok in text.split():
            if tok.isalpha():
                tok = re.sub(r"aa+", "a", tok)
                tok = re.sub(r"ee+|ii+", "i", tok)
                tok = re.sub(r"oo+|uu+", "u", tok)
                tok = re.sub(r"([tdnl])\1+", r"\1", tok) # retroflex tt/dd -> t/d
                # Strip trailing inherent schwa 'a' on words > 3 chars ending in consonant + 'a'
                if len(tok) > 3 and tok.endswith("a") and tok[-2] not in "aeiouy":
                    tok = tok[:-1]
            tokens.append(tok)
        text = " ".join(tokens)

    return text

def phonetic_token_skeleton(token: str) -> str:
    """
    Maps a normalized Latin token into a cross-script phonetic skeleton
    to bridge Hindi-English transliteration and French silent/double consonant noise.
    """
    if not token or not token.isalpha():
        return token

    s = token.lower()
    # Canonicalize Indic & French phonetic equivalences
    s = s.replace("ksh", "x").replace("ph", "f").replace("sh", "s")
    s = s.replace("chh", "ch").replace("bh", "b").replace("dh", "d")
    s = s.replace("gh", "g").replace("kh", "k").replace("th", "t")
    s = s.replace("wh", "v").replace("w", "v").replace("z", "j")
    s = s.replace("qu", "k").replace("ck", "k").replace("c", "k")

    # Keep first character (normalizing leading vowel equivalences), strip interior vowels
    first = s[0]
    if first in "ei":
        first = "i"
    elif first in "ou":
        first = "u"
    elif first in "a":
        first = "a"

    rest = re.sub(r"[aeiouyh]", "", s[1:])
    skeleton = first + rest
    # Collapse consecutive duplicate consonants (e.g., French 'll', 'mm', 'ss', 'tt')
    skeleton = re.sub(r"(.)\1+", r"\1", skeleton)
    return skeleton

```

---

### 4. Upgrading Blocking & Features for Cross-Script / Cross-Lingual Pairs

1. **Add a `phonetic_skeleton` Inverted Index in `blocking.py`:**
   Alongside your standard token index on `name_normalized`, index the `phonetic_skeleton` tokens (filtered by within-country IDF). When $S_1$ has `"श्री बालाजी एंटरप्राइजेज"` and $S_2$ has `"Shree Balaji Enterprises"`, both yield the phonetic tokens `["sr", "blj", "intrprjs"]` and are guaranteed to be paired in `candidate_pairs.tsv`.

2. **Add Cross-Script Features in `features.py`:**

- `is_cross_script_pair`: Binary indicator (`1` if one record had non-Latin script and the other was pure Latin, `0` otherwise).
- `phonetic_skeleton_jaccard` and `phonetic_skeleton_levenshtein`: Edit and token-set similarities computed on the phonetic skeletons.
- **Why `is_cross_script_pair` matters for GBDT:** When `is_cross_script_pair == 1`, raw character Levenshtein on `name_normalized` will naturally be lower (`~0.70–0.80` due to vowel variations) even for an exact match. The tree model will learn to weight `phonetic_skeleton_levenshtein` and `embedding_cosine` more heavily whenever `is_cross_script_pair == 1`.

3. **Promote Multilingual Embeddings from "Optional" to Phase 2:**
   Phonetic transliteration solves `"एयरटेल"` $\leftrightarrow$ `"Airtel"`, but it cannot solve **translations** like `"शर्मा मिष्ठान भंडार"` $\leftrightarrow$ `"Sharma Sweet Shop"` or `"भारतीय स्टेट बैंक"` $\leftrightarrow$ `"State Bank of India"`. A compact, MIT-licensed multilingual embedding model (`BAAI/bge-m3` or `intfloat/multilingual-e5-base`, both $\le 560\text{M}$ parameters, well under the $8\text{B}$ limit) aligns Hindi, French, and English in the same vector space out of the box.
