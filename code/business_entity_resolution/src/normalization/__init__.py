from .normalizer import ADDR_FIELDS, NAME_FIELDS, normalize_address, normalize_name, normalize_records
from .text import fold, phonetic_key
from .transliterate import learn_lexicon, load_lexicon, save_lexicon

__all__ = [
    "ADDR_FIELDS",
    "NAME_FIELDS",
    "fold",
    "learn_lexicon",
    "load_lexicon",
    "normalize_address",
    "normalize_name",
    "normalize_records",
    "phonetic_key",
    "save_lexicon",
]
