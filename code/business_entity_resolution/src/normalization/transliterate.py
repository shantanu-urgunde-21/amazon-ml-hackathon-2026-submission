"""Learn an Indic-word -> English-word lexicon from the training ground truth.

Why: Indic records in S2/S3 are mostly English words *written* in an Indic
script ("प्राइवेट लिमिटेड" = "private limited"). Rule-based romanization gives
"praivet limited"; the matched S1 record tells us the real word.

How: for each ground-truth pair (S1 record, Indic S2/S3 record)
  - names: if both names have the same number of tokens, align them by position
  - addresses: align comma-separated components with the same token count
    (this is where state names like "महाराष्ट्र" -> "maharashtra" come from)
A mapping is kept when it is seen >= min_count times and is the dominant
target (>= min_purity of that word's alignments).

When validating, pass only the training fold's ground truth, otherwise the
lexicon leaks labels from the validation fold.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .text import INDIC_RE, fold


class _Placeholder:
    """Stand-in lexicon for fold(): swaps each Indic word for a unique marker
    token so it survives folding and can be aligned by position."""

    def __init__(self):
        self.words = {}

    def get(self, word):
        key = "qxq" + "".join(chr(97 + int(d)) for d in str(len(self.words))) + "qxq"
        self.words[key] = word
        return key


def _align(indic_raw: str, english_raw: str, counts):
    ph = _Placeholder()
    toks = fold(indic_raw, ph).split()
    eng = fold(english_raw).split()
    if not ph.words or len(toks) != len(eng):
        return
    for t, e in zip(toks, eng):
        if t in ph.words and not e.isdigit():
            counts[ph.words[t]][e] += 1


def learn_lexicon(
    pairs: pd.DataFrame,
    min_count: int = 3,
    min_purity: float = 0.6,
    min_cooccurrence: float = 0.8,
) -> dict:
    """pairs: DataFrame with raw columns name1, addr1 (S1) and name2, addr2 (S2/S3).

    Names align one-to-one, so a word's dominant target must hold `min_purity`
    of its alignments. An address component is aligned against every S1
    component of the same length (city, locality, state ...), so there the
    target must instead co-occur in `min_cooccurrence` of the pairs where the
    Indic word appears: the state is always there, the cities vary.
    """
    name_counts = defaultdict(Counter)
    addr_counts = defaultdict(Counter)
    addr_seen = Counter()
    for n1, a1, n2, a2 in zip(pairs["name1"], pairs["addr1"], pairs["name2"], pairs["addr2"]):
        if INDIC_RE.search(n2):
            _align(n2, n1, name_counts)
        if INDIC_RE.search(a2):
            s1_comps = [c for c in a1.split(",") if c.strip()]
            s1_lens = [len(fold(c).split()) for c in s1_comps]
            for comp in a2.split(","):
                if not INDIC_RE.search(comp):
                    continue
                ph = _Placeholder()
                toks = fold(comp, ph).split()
                pair_counts = defaultdict(Counter)
                if len(toks) == 1 and toks[0] in ph.words:
                    # one Indic word may be several English words: "தமிழ்நாடு" = "tamil nadu"
                    for sc in s1_comps:
                        pair_counts[ph.words[toks[0]]][fold(sc)] += 1
                else:
                    for sc, ln in zip(s1_comps, s1_lens):
                        if ln == len(toks):
                            _align(comp, sc, pair_counts)
                for word in set(ph.words.values()):
                    addr_seen[word] += 1
                    for eng in pair_counts.get(word, {}):
                        addr_counts[word][eng] += 1  # once per pair

    lexicon = {}
    for word, c in addr_counts.items():
        eng, top = c.most_common(1)[0]
        if top >= min_count and top / addr_seen[word] >= min_cooccurrence:
            lexicon[word] = eng
    for word, c in name_counts.items():  # names win on conflict
        eng, top = c.most_common(1)[0]
        if top >= min_count and top / sum(c.values()) >= min_purity:
            lexicon[word] = eng
    return lexicon


def save_lexicon(lexicon: dict, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(lexicon, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")


def load_lexicon(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}

