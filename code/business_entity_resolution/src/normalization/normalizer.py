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

# Extended Unicode ranges for Indic scripts:
# Devanagari (\u0900-\u097F), Bengali (\u0980-\u09FF), Gurmukhi (\u0A00-\u0A7F),
# Gujarati (\u0A80-\u0AFF), Odia (\u0B00-\u0B7F), Tamil (\u0B80-\u0BFF),
# Telugu (\u0C00-\u0C7F), Kannada (\u0C80-\u0CFF), Malayalam (\u0D00-\u0D7F)
INDIC_SCRIPTS_RE = re.compile(r"[\u0900-\u0D7F]")
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

# Domain / URL cleaning pattern
URL_RE = re.compile(
    r"\b(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|in|org|net|co|fr|gov|io|biz|info)\b",
    re.IGNORECASE
)

# Street designator keywords for US, India, France
STREET_DESIGNATORS = (
    r"(?:road|rd|street|st|avenue|ave|drive|dr|lane|ln|blvd|boulevard|way|"
    r"court|ct|place|pl|circle|cir|highway|hwy|rue|place|chemin|all[eé]e)"
)

STREET_PATTERN_US = re.compile(
    r"\b(\d+)\s+([a-z0-9\s]{2,25}?)\s+" + STREET_DESIGNATORS + r"\b",
    re.IGNORECASE
)

# State and Administrative Region Dictionaries (US, India, France)
# NOTE: Pre-compiled domain dictionaries for regional entity disambiguation.
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia"
}

INDIA_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland",
    "od": "odisha", "or": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim",
    "tn": "tamil nadu", "tg": "telangana", "ts": "telangana", "tr": "tripura", "up": "uttar pradesh",
    "uk": "uttarakhand", "ua": "uttarakhand", "wb": "west bengal", "dl": "delhi"
}

FRANCE_REGIONS = {
    "idf": "ile de france", "ara": "auvergne rhone alpes", "bfc": "bourgogne franche comte",
    "bre": "bretagne", "cvl": "centre val de loire", "cor": "corse", "ges": "grand est",
    "hdf": "hauts de france", "nor": "normandie", "naq": "nouvelle aquitaine", "occ": "occitanie",
    "pdl": "pays de la loire", "paca": "provence alpes cote d azur"
}

FRANCE_MAJOR_CITIES = {
    "paris": "fr_idf", "lyon": "fr_ara", "marseille": "fr_paca", "toulouse": "fr_occ",
    "nice": "fr_paca", "nantes": "fr_pdl", "montpellier": "fr_occ", "strasbourg": "fr_ges",
    "bordeaux": "fr_naq", "lille": "fr_hdf", "rennes": "fr_bre", "toulon": "fr_paca",
    "grenoble": "fr_ara", "dijon": "fr_bfc", "angers": "fr_pdl", "nimes": "fr_occ",
    "aix en provence": "fr_paca", "brest": "fr_bre", "le mans": "fr_pdl", "amiens": "fr_hdf",
    "tours": "fr_cvl", "limoges": "fr_naq", "clermont ferrand": "fr_ara", "besancon": "fr_bfc"
}

# 1. Full state and region names (unambiguous, length >= 4)
FULL_NAMES_MAP: Dict[str, str] = {}
for code, name in US_STATES.items():
    FULL_NAMES_MAP[name] = code
for code, name in INDIA_STATES.items():
    canonical = "in_tg" if code in ("tg", "ts") else ("in_od" if code in ("od", "or") else ("in_uk" if code in ("uk", "ua") else f"in_{code}"))
    FULL_NAMES_MAP[name] = canonical
for code, name in FRANCE_REGIONS.items():
    FULL_NAMES_MAP[name] = f"fr_{code}"
for city, region_code in FRANCE_MAJOR_CITIES.items():
    FULL_NAMES_MAP[city] = region_code

FULL_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(FULL_NAMES_MAP.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE
)

# 2. Positional 2-letter codes: preceded by comma, followed by optional zip or end of address
CODE_PATTERN = re.compile(r",\s*([a-zA-Z]{2})(?:\s*,|\s+\d{5,6}|\s*$)", re.IGNORECASE)
STOP_WORDS_2 = {"in", "or", "me", "as", "so", "to", "at", "by", "of", "no", "is", "am", "it", "on", "he"}





def has_non_latin_script(text: str) -> bool:
    """Detects whether text contains Indic or other non-Latin scripts."""
    if not text:
        return False
    return bool(INDIC_SCRIPTS_RE.search(text)) or any(ord(c) > 0x024F for c in text)


def clean_text(text: str) -> str:
    """Backward-compatible alias for transliterate_and_fold."""
    return transliterate_and_fold(text)


def transliterate_and_fold(text: str) -> str:
    """
    Converts Indic scripts to ASCII and strips French/Latin diacritics
    while preserving token boundaries across French elisions (l', d') and hyphens.
    Also unpacks domains (e.g. sjacevendome.com -> sjacevendome) and handles tags.
    """
    if not text or not isinstance(text, str):
        return ""

    # Unpack domain names and strip leading hashtags / mentions
    text = URL_RE.sub(r" \1 ", text)
    text = re.sub(r"^[#@]+", "", text)

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
    to bridge Hindi/Indic-English transliterations and French silent/double consonant variations.
    """
    if not token or not token.isalpha():
        return token

    s = token.lower()
    s = s.replace("ksh", "x").replace("ph", "f").replace("sh", "s")
    s = s.replace("chh", "ch").replace("bh", "b").replace("dh", "d")
    s = s.replace("gh", "g").replace("kh", "k").replace("th", "t")
    s = s.replace("wh", "v").replace("w", "v").replace("z", "j")
    s = s.replace("qu", "k").replace("ck", "k").replace("c", "k")
    s = s.replace("ks", "x")
    s = s.replace("np", "mp").replace("nb", "mb")

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


def extract_state_region(address_text: str) -> str:
    """Extracts standardized US or Indian state code safely from address without matching stop words."""
    if not address_text:
        return ""
    # Check full names first
    m_full = FULL_NAME_PATTERN.findall(address_text.lower())
    if m_full:
        return FULL_NAMES_MAP[m_full[-1]]

    # Check positional 2-letter code preceded by comma
    m_code = CODE_PATTERN.search(address_text)
    if m_code:
        code = m_code.group(1).lower()
        if code not in STOP_WORDS_2:
            if code in US_STATES:
                return code
            if code in INDIA_STATES:
                return "in_tg" if code in ("tg", "ts") else ("in_od" if code in ("od", "or") else ("in_uk" if code in ("uk", "ua") else f"in_{code}"))
    return ""



def extract_primary_street_info(address_text: str) -> Tuple[str, str]:
    """
    Extracts primary street number and street name from address.
    Returns: (street_number, street_name)
    """
    if not address_text:
        return "", ""
    clean = address_text.lower().replace(",", " ").replace("/", " ")
    clean = WHITESPACE_RE.sub(" ", clean).strip()

    m = STREET_PATTERN_US.search(clean)
    if m:
        num = m.group(1).strip()
        street = m.group(2).strip()
        return num, street

    # Fallback to leading numeric token
    m_num = re.search(r"\b(\d{1,6})\b", clean)
    num = m_num.group(1) if m_num else ""
    return num, ""


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies multilingual normalization and structured slot extraction to DataFrame.
    Adds:
    - norm_name, norm_address, name_tokens, address_tokens
    - numeric_tokens, phonetic_tokens, has_non_latin
    - postal_code, building_number, unit_slot
    - state_region, primary_street_num, street_name
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
    df["state_region"] = raw_addrs.map(extract_state_region)

    street_info = [extract_primary_street_info(a) for a in raw_addrs]
    df["primary_street_num"] = [info[0] for info in street_info]
    df["street_name"] = [info[1] for info in street_info]

    return df

