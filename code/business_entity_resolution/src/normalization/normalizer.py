"""
Normalization and Structured Slot Extraction module for Business Entity Resolution.
Extracts:
1. Multilingual text folding (French ligatures, apostrophes, Devanagari transliteration)
2. Phonetic consonant skeletons
3. Structured address slots:
   - postal_code (5-digit US/FR, 6-digit Indian PIN)
   - building_number (e.g. 14, 14B, 14-bis, plot 12)
   - sub_unit (apt, suite, unit, shop, floor)
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from unidecode import unidecode

LIGATURE_MAP = str.maketrans({
    "œ": "oe", "Œ": "oe",
    "æ": "ae", "Æ": "ae",
    "ß": "ss", "&": " and ",
})

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
APOSTROPHE_HYPHEN_RE = re.compile(r"['’`\-–—/.,()\[\]{}:;\"!|+]+")
CLEAN_CHARS_RE = re.compile(r"[^a-z0-9\s]")
WHITESPACE_RE = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\b\d{2,}\b")

# Structured address regexes
# India: 6 digits (optionally space separated 3+3)
# US: 5 digits (optionally +4)
# France: 5 digits (starting with 01-98 or 20)
POSTAL_CODE_RE = re.compile(r"\b(?:\d{6}|\d{5}(?:-\d{4})?)\b")

# Building / Plot numbers: e.g., "plot 14", "plot no. 14b", "no. 23", "h.no 102", "14 bis"
BUILDING_NUM_RE = re.compile(
    r"\b(?:plot(?:\s+no\.?)?|h\.?\s*no\.?|house\s+no\.?|no\.?|bldg\.?|building)?\s*#?\s*(\d+[a-z]?(?:\s*(?:bis|ter))?)\b",
    re.IGNORECASE
)

# Sub-unit slots: suite 400, apt 3b, floor 2, shop g-4
UNIT_RE = re.compile(
    r"\b(?:suite|ste|apt|apartment|unit|shop|shp|fl|floor|room|rm)\s*#?\s*([a-z0-9\-]+)\b",
    re.IGNORECASE
)


def has_non_latin_script(text: str) -> bool:
    """Detects whether text contains Devanagari or other non-Latin scripts."""
    if not text:
        return False
    return bool(DEVANAGARI_RE.search(text)) or any(ord(c) > 0x024F for c in text)


def clean_text(text: str) -> str:
    """Backward-compatible alias for transliterate_and_fold."""
    return transliterate_and_fold(text)


def transliterate_and_fold(text: str) -> str:
    """
    Converts Devanagari/Indic scripts to ASCII and strips French/Latin diacritics
    while preserving token boundaries across French elisions (l', d') and hyphens.
    """
    if not text or not isinstance(text, str):
        return ""

    is_non_latin = has_non_latin_script(text)
    text = text.translate(LIGATURE_MAP)

    if is_non_latin:
        text = unidecode(text)
    else:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))

    text = text.lower()
    text = APOSTROPHE_HYPHEN_RE.sub(" ", text)
    text = CLEAN_CHARS_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    if is_non_latin:
        tokens = []
        for tok in text.split():
            if tok.isalpha():
                tok = re.sub(r"aa+", "a", tok)
                tok = re.sub(r"ee+|ii+", "i", tok)
                tok = re.sub(r"oo+|uu+", "u", tok)
                tok = re.sub(r"([tdnl])\1+", r"\1", tok)
                if len(tok) > 3 and tok.endswith("a") and tok[-2] not in "aeiouy":
                    tok = tok[:-1]
            tokens.append(tok)
        text = " ".join(tokens)

    return text


def phonetic_token_skeleton(token: str) -> str:
    """
    Maps a normalized Latin token into a cross-script phonetic skeleton
    to bridge Hindi-English transliterations and French silent/double consonant variations.
    """
    if not token or not token.isalpha():
        return token

    s = token.lower()
    s = s.replace("ksh", "x").replace("ph", "f").replace("sh", "s")
    s = s.replace("chh", "ch").replace("bh", "b").replace("dh", "d")
    s = s.replace("gh", "g").replace("kh", "k").replace("th", "t")
    s = s.replace("wh", "v").replace("w", "v").replace("z", "j")
    s = s.replace("qu", "k").replace("ck", "k").replace("c", "k")

    first = s[0]
    if first in "ei":
        first = "i"
    elif first in "ou":
        first = "u"
    elif first in "a":
        first = "a"

    rest = re.sub(r"[aeiouyh]", "", s[1:])
    skeleton = first + rest
    skeleton = re.sub(r"(.)\1+", r"\1", skeleton)
    return skeleton


def extract_tokens(clean_s: str, min_len: int = 2) -> List[str]:
    """Extracts non-empty tokens with minimum length."""
    if not clean_s:
        return []
    return [t for t in clean_s.split(" ") if len(t) >= min_len]


def extract_numeric_tokens(raw_or_clean: str) -> Set[str]:
    """Extracts numeric strings of length 2 or more."""
    if not raw_or_clean:
        return set()
    return set(RE_NUMBERS.findall(raw_or_clean))


def extract_postal_code(address_text: str) -> str:
    """Extracts the first matching postal/PIN code."""
    if not address_text:
        return ""
    m = POSTAL_CODE_RE.search(address_text)
    if m:
        # Normalize 5-digit zip by dropping +4 suffix if present
        code = m.group(0).split("-")[0]
        return code
    return ""


def extract_building_number(address_text: str) -> str:
    """Extracts building or plot number."""
    if not address_text:
        return ""
    m = BUILDING_NUM_RE.search(address_text)
    if m:
        val = m.group(1).lower().replace(" ", "")
        return val
    return ""


def extract_unit(address_text: str) -> str:
    """Extracts unit, suite, apartment, or shop number."""
    if not address_text:
        return ""
    m = UNIT_RE.search(address_text)
    if m:
        return m.group(1).lower().strip()
    return ""


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies multilingual normalization and structured slot extraction to DataFrame.
    Adds:
    - norm_name, norm_address, name_tokens, address_tokens
    - numeric_tokens, phonetic_tokens, has_non_latin
    - postal_code, building_number, unit_slot
    """
    df = df.copy()

    raw_names = df["business_name"].astype(str)
    raw_addrs = df["business_address"].astype(str)

    df["has_non_latin"] = [
        has_non_latin_script(n) or has_non_latin_script(a)
        for n, a in zip(raw_names, raw_addrs)
    ]

    df["norm_name"] = raw_names.map(transliterate_and_fold)
    df["norm_address"] = raw_addrs.map(transliterate_and_fold)

    df["name_tokens"] = df["norm_name"].map(lambda s: extract_tokens(s, min_len=2))
    df["address_tokens"] = df["norm_address"].map(lambda s: extract_tokens(s, min_len=2))

    df["phonetic_tokens"] = df["name_tokens"].map(
        lambda toks: [phonetic_token_skeleton(t) for t in toks if len(t) >= 3]
    )

    df["numeric_tokens"] = raw_addrs.map(extract_numeric_tokens)

    # Structured slots
    df["postal_code"] = raw_addrs.map(extract_postal_code)
    df["building_number"] = raw_addrs.map(extract_building_number)
    df["unit_slot"] = raw_addrs.map(extract_unit)

    return df
