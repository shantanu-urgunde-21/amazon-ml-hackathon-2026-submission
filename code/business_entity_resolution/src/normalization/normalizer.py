"""Record normalization: raw name/address -> cleaned, typed fields.

Name fields
  name_norm     folded full name, legal forms canonicalized ("pvt ltd" -> pvt_ltd)
  name_core     name_norm without legal forms, honorifics and articles
  name_alias    core of the part after "DBA / formerly / nee / | www..." ("" if none)
  name_compact  name_core without spaces (matches domains like "lifeinvestments.com")
  name_phon     phonetic key of each core token
  legal         canonical legal forms found, space separated ("llc", "pvt_ltd", "sarl")
  has_indic     name or address contained Indic script

Address fields
  addr_norm     folded address without state, PO box and locality noise, abbreviations canonical
  state         canonical state/region code ("us_nc", "in_mh", "fr_hdf"), "" if not found
  house_num     first number of the first component containing a digit, leading zeros stripped
  addr_nums     all number codes, separators removed ("C-2-08" -> "c208", "4/792" -> "4792")

Everything is computed once per unique (string, country) and in parallel.
"""

import os
import re
from multiprocessing import Pool

import pandas as pd

from .lexicons import (
    ADDR_ABBREV, ADDR_ABBREV_COMMON, ALIAS_PATTERN, LEGAL_PHRASES, LEGAL_TOKENS,
    LEGAL_TOKENS_COMMON, LOCALITY_NOISE, NAME_STOPWORDS, NAME_STOPWORDS_COMMON,
    STATE_LOOKUP, STATE_LOOKUP_ANY,
)
from .text import INDIC_RE, fold, phonetic_key

# Bump when normalization output changes: it is part of the cache key.
NORMALIZER_VERSION = 2

NAME_FIELDS = ["name_norm", "name_core", "name_alias", "name_compact", "name_phon", "legal"]
ADDR_FIELDS = ["addr_norm", "state", "house_num", "addr_nums"]

_ALL_LEGAL = {k: v for d in LEGAL_TOKENS.values() for k, v in d.items()}
_ALL_STOP = set().union(*NAME_STOPWORDS.values())
_ALL_ABBREV = {k: v for d in ADDR_ABBREV.values() for k, v in d.items() if len(k) > 2}

DOMAIN_SUFFIX_RE = re.compile(r"\s*\|\s*((?:www\.)?\S+?\.(?:com|in|net|org|fr|co\.in))\S*\s*$", re.I)
DOMAIN_ONLY_RE = re.compile(r"^(?:www\.)?(\S+?)\.(?:com|in|net|org|fr|co\.in)$", re.I)
FUSED_COM_RE = re.compile(r"^([A-Z]{7,})COM$")
TRAILING_TAG_RE = re.compile(r"\s+#\d+\s*$")
ALIAS_RE = re.compile(ALIAS_PATTERN)
ORDINAL_RE = re.compile(r"^0*(\d+)(?:st|nd|rd|th|eme|er|e)?$")  # 152nd / 2eme / 004512
PO_BOX_RE = re.compile(r"\bp\s*o\s*box\s*\w*|\bpmb\s*\d+|\bbp\s*\d+|\bcedex\s*\d*")
NULL_TOKENS = {"null", "none", "nan"}
PHONE_RE = re.compile(r"[\s\-]*\+?\d[\d\s\-]{6,}\d\s*$")  # "TITAN LLC CENTER - 6424028144"
NUM_CODE_RE = re.compile(r"[A-Za-z]?\d[\dA-Za-z/\-#]*")
FIRST_NUM_RE = re.compile(r"\d+")

# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------


def _legal_and_core(tokens, country):
    legal_map = {**LEGAL_TOKENS_COMMON, **LEGAL_TOKENS.get(country, _ALL_LEGAL)}
    stop = NAME_STOPWORDS_COMMON | NAME_STOPWORDS.get(country, _ALL_STOP)
    norm, core, legal = [], [], []
    i = 0
    while i < len(tokens):
        for n in (3, 2):
            phrase = tuple(tokens[i:i + n])
            if len(phrase) == n and phrase in LEGAL_PHRASES:
                code = LEGAL_PHRASES[phrase]
                if code:
                    norm.append(code)
                    legal.append(code)
                i += n
                break
        else:
            t = tokens[i]
            if t in legal_map:
                norm.append(legal_map[t])
                legal.append(legal_map[t])
            else:
                norm.append(t)
                if t not in stop:
                    core.append(t)
            i += 1
    return norm, core, legal


def normalize_name(raw: str, country: str, lexicon: dict) -> tuple:
    raw = raw or ""
    alias = ""
    m = DOMAIN_SUFFIX_RE.search(raw)
    if m:  # "Name | www.name.com": keep the domain body as alias
        alias = DOMAIN_ONLY_RE.sub(r"\1", m.group(1))
        raw = raw[:m.start()]
    raw = DOMAIN_ONLY_RE.sub(r"\1", raw.strip())
    raw = FUSED_COM_RE.sub(r"\1", raw)
    raw = TRAILING_TAG_RE.sub("", raw)
    raw = PHONE_RE.sub("", raw)

    folded = fold(raw, lexicon)
    parts = [p.strip() for p in ALIAS_RE.split(folded) if p.strip()]
    folded = " ".join(parts)
    norm, core, legal = _legal_and_core(folded.split(), country)
    if len(parts) > 1:
        alias = " ".join(_legal_and_core(parts[-1].split(), country)[1])
    elif alias:
        alias = " ".join(_legal_and_core(fold(alias).split(), country)[1])


    core_s = " ".join(core) if core else " ".join(norm)
    return (
        " ".join(norm),
        core_s,
        alias,
        core_s.replace(" ", ""),
        " ".join(phonetic_key(t) for t in core_s.split()),
        " ".join(sorted(set(legal))),
    )


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------


def normalize_address(raw: str, country: str, lexicon: dict) -> tuple:
    raw = raw or ""
    states = STATE_LOOKUP.get(country, STATE_LOOKUP_ANY)
    abbrev = {**ADDR_ABBREV_COMMON, **ADDR_ABBREV.get(country, _ALL_ABBREV)}

    comps = [c for c in raw.split(",") if c.strip()]
    folded = [fold(c, lexicon) for c in comps]

    state, state_i = "", -1
    for i in range(len(folded) - 1, -1, -1):  # state is usually last, but order varies
        f = folded[i]
        if f in states:
            state, state_i = states[f], i
            break
    if not state:  # state glued into a component: "..., new delhi delhi"
        for i in range(len(folded) - 1, -1, -1):
            for n in (3, 2, 1):
                tail = " ".join(folded[i].split()[-n:])
                if len(tail) > 3 and tail in states and tail != folded[i]:
                    state = states[tail]
                    folded[i] = folded[i][: -len(tail)].strip()
                    break
            if state:
                break

    tokens, house_num, nums = [], "", []
    for i, (c, f) in enumerate(zip(comps, folded)):
        if i == state_i:
            continue
        f = PO_BOX_RE.sub(" ", f)
        c = PO_BOX_RE.sub(" ", c.lower())
        for code in NUM_CODE_RE.findall(c):
            code = ORDINAL_RE.sub(r"\1", re.sub(r"[/\-#]", "", code)).lstrip("0")
            if code:
                nums.append(code)
        if not house_num:
            m = FIRST_NUM_RE.search(c)
            if m:  # "506-510" -> 506;  India "C-2-08" -> 208 (separators are noise there)
                house_num = m.group(0).lstrip("0") or "0"
                if country == "India" and nums:
                    house_num = re.sub(r"\D", "", nums[0]) or house_num
        for t in f.split():
            m = ORDINAL_RE.match(t)
            t = m.group(1) if m else abbrev.get(t, t)
            if t and t not in LOCALITY_NOISE and t not in NULL_TOKENS:
                tokens.append(t)

    return " ".join(tokens), state, house_num, " ".join(dict.fromkeys(nums))


# --------------------------------------------------------------------------
# DataFrame-level entry point
# --------------------------------------------------------------------------

_LEXICON = {}


def _init(lexicon):
    global _LEXICON
    _LEXICON = lexicon


def _names_chunk(items):
    return [normalize_name(s, c, _LEXICON) for s, c in items]


def _addrs_chunk(items):
    return [normalize_address(s, c, _LEXICON) for s, c in items]


def _parallel_map(fn, items, lexicon, workers):
    if len(items) < 20_000 or workers <= 1:
        _init(lexicon)
        return fn(items)
    size = max(5_000, len(items) // (workers * 8))
    chunks = [items[i:i + size] for i in range(0, len(items), size)]
    with Pool(workers, initializer=_init, initargs=(lexicon,)) as pool:
        out = []
        for part in pool.imap(fn, chunks):
            out.extend(part)
    return out


def normalize_records(df: pd.DataFrame, lexicon: dict | None = None, workers: int | None = None) -> pd.DataFrame:
    """Adds NAME_FIELDS + ADDR_FIELDS + has_indic to a raw source frame."""
    lexicon = lexicon or {}
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    out = df  # columns are added in place (callers pass their own frame)

    for col, fn, fields in [
        ("business_name", _names_chunk, NAME_FIELDS),
        ("business_address", _addrs_chunk, ADDR_FIELDS),
    ]:
        keys = pd.MultiIndex.from_arrays([out[col].fillna(""), out["country"].fillna("")])
        uniq = keys.unique()
        results = _parallel_map(fn, list(uniq), lexicon, workers)
        table = pd.DataFrame(results, columns=fields, index=uniq)
        pos = uniq.get_indexer(keys)
        for f in fields:
            out[f] = table[f].to_numpy()[pos]

    out["has_indic"] = (
        out["business_name"].fillna("").str.contains(INDIC_RE.pattern)
        | out["business_address"].fillna("").str.contains(INDIC_RE.pattern)
    )
    return out
