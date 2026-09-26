"""Script-level text folding shared by name and address normalization.

fold(text, lexicon) -> lowercase ASCII tokens separated by single spaces.

Indic-script words (S2/S3 only; S1 is always Latin) are converted by:
  1. the learned lexicon (Indic word -> English word, see transliterate.py)
  2. anyascii romanization as fallback (lossy: "सर्विसेज" -> "srvisej")
"""

import re
import unicodedata

from anyascii import anyascii

# Devanagari (U+0900) .. Malayalam (U+0D7F): Hindi, Marathi, Bengali, Gurmukhi,
# Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam.
INDIC_RE = re.compile(r"[ऀ-ൿ]")
INDIC_WORD_RE = re.compile(r"[ऀ-ൿ‌‍]+")
ZERO_WIDTH_RE = re.compile(r"[​-‏⁠﻿­]")

PRE_MAP = str.maketrans({
    "œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE", "ß": "ss",
    "’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'",
    "–": "-", "—": "-", "‐": "-", "‑": "-",
    "°": "o", "º": "o", "ª": "a",  # French "N°" / "Nº" -> "No"
    "&": " and ", "@": " ",
})

DOTTED_ACRONYM_RE = re.compile(r"\b(?:[a-z]\.){2,}(?:[a-z]\b)?")  # l.l.c. -> llc
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
# OCR-style digit-for-letter swaps inside words: "Appare1s", "vippub1ic"
OCR_DIGIT_RE = re.compile(r"(?<=[a-z])[01](?=[a-z])")
OCR_MAP = str.maketrans({"0": "o", "1": "l"})


def romanize_indic(text: str, lexicon: dict | None = None) -> str:
    lexicon = lexicon or {}

    def repl(m):
        word = m.group(0).replace("‌", "").replace("‍", "")
        eng = lexicon.get(word)
        return f" {eng} " if eng else f" {anyascii(word)} "

    return INDIC_WORD_RE.sub(repl, text)


def fix_ocr_digits(token: str) -> str:
    if token.isalpha() or token.isdigit() or sum(c.isalpha() for c in token) < 3:
        return token
    return OCR_DIGIT_RE.sub(lambda m: m.group(0).translate(OCR_MAP), token)


def fold(text: str, lexicon: dict | None = None) -> str:
    """Raw string -> lowercase ASCII tokens separated by single spaces."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub("", text).translate(PRE_MAP)
    if INDIC_RE.search(text):
        text = romanize_indic(text, lexicon)
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = anyascii(text)
    text = text.lower()
    text = DOTTED_ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), text)
    text = text.replace("'", "")
    text = NON_ALNUM_RE.sub(" ", text)
    return " ".join(fix_ocr_digits(t) for t in text.split())


# --------------------------------------------------------------------------
# Phonetic key: bridges typos and lossy romanization.
#   "marketing" and anyascii "marketimg" -> "mrktn"
#   "shree" / "sri" / "shri" -> "sr",  "balaji" -> "blj"
# --------------------------------------------------------------------------

_PHONETIC_RULES = [
    (re.compile(r"(.)\1+"), r"\1"),
    (re.compile(r"ph"), "f"),
    (re.compile(r"m(?=[kgtdcjsbp])"), "n"),     # anusvara written as m before consonants
    (re.compile(r"ksh|x"), "ks"),
    (re.compile(r"ck|q|c(?=[aouklrt]|$)"), "k"),
    (re.compile(r"c"), "s"),
    (re.compile(r"z"), "j"),
    (re.compile(r"w"), "v"),
    (re.compile(r"([kgcjtdpbs])h"), r"\1"),     # aspirates: bh, dh, th, kh, sh
    (re.compile(r"(?<=n)g$"), ""),              # -ing -> -in
]
_VOWELS_RE = re.compile(r"[aeiouyh]")
_DOUBLE_RE = re.compile(r"(.)\1+")


def phonetic_key(token: str) -> str:
    if not token or not token.isalpha():
        return token
    s = token
    for pat, rep in _PHONETIC_RULES:
        s = pat.sub(rep, s)
    return _DOUBLE_RE.sub(r"\1", s[0] + _VOWELS_RE.sub("", s[1:]))
