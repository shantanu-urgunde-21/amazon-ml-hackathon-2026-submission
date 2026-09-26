"""Hand-curated word lists used by the normalizer.

Keys are in *folded* form (lowercase ASCII, punctuation removed), because every
lookup happens after `text.fold`.

Country-specific tables are looked up by the record's `country` label. An
unknown country falls back to the union of all tables, so a new country still
gets generic cleanup instead of being dropped (the country set is open).
"""

# --------------------------------------------------------------------------
# Business names
# --------------------------------------------------------------------------

# Multi-token legal forms, matched on the token sequence before single tokens.
LEGAL_PHRASES = {
    ("private", "limited"): "pvt_ltd",
    ("pvt", "limited"): "pvt_ltd",
    ("private", "ltd"): "pvt_ltd",
    ("pvt", "ltd"): "pvt_ltd",
    ("limited", "liability", "company"): "llc",
    ("limited", "liability", "partnership"): "llp",
    ("doing", "business", "as"): "",
}

LEGAL_TOKENS_COMMON = {
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt", "pvte": "pvt",
    "llp": "llp", "llc": "llc", "plc": "plc", "company": "co", "co": "co",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
}
LEGAL_TOKENS = {
    "US": {"pc": "pc", "pllc": "pllc", "lp": "lp", "pa": "pa"},
    "India": {"opc": "opc"},
    "France": {
        "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
        "sci": "sci", "snc": "snc", "selarl": "selarl", "scop": "scop", "scm": "scm",
        "scp": "scp", "gie": "gie",
    },
}

# Tokens that carry no identity (honorifics, articles). Removed from name_core.
NAME_STOPWORDS_COMMON = {"the", "and", "of"}
NAME_STOPWORDS = {
    "US": {"mr", "mrs", "ms", "dr"},
    "India": {"sri", "shri", "shree", "smt", "mr", "mrs", "ms", "dr", "m", "s"},
    "France": {"et", "le", "la", "les", "l", "de", "des", "du", "d", "france"},
}

# "X DBA Y", "X formerly Y", "X nee Y": both sides are kept as name variants.
ALIAS_PATTERN = (
    r"\b(?:d\s*b\s*a|doing business as|formerly known as|formerly|f\s*k\s*a|"
    r"a\s*k\s*a|nee|trading as|t\s*a)\b"
)

# --------------------------------------------------------------------------
# Addresses: states / regions
# --------------------------------------------------------------------------

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska",
    "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia", "pr": "puerto rico",
}

# code -> accepted surface forms (first is canonical)
INDIA_STATES = {
    "an": ["andaman and nicobar islands", "andaman and nicobar"],
    "ap": ["andhra pradesh"],
    "ar": ["arunachal pradesh"],
    "as": ["assam"],
    "br": ["bihar"],
    "ch": ["chandigarh"],
    "cg": ["chhattisgarh", "chattisgarh", "ct"],
    "dn": ["dadra and nagar haveli and daman and diu", "dadra and nagar haveli",
           "daman and diu", "dd"],
    "dl": ["delhi", "nct of delhi"],
    "ga": ["goa"],
    "gj": ["gujarat"],
    "hr": ["haryana"],
    "hp": ["himachal pradesh"],
    "jk": ["jammu and kashmir", "jammu kashmir"],
    "jh": ["jharkhand"],
    "ka": ["karnataka"],
    "kl": ["kerala", "keralam"],
    "la": ["ladakh"],
    "ld": ["lakshadweep"],
    "mp": ["madhya pradesh"],
    "mh": ["maharashtra"],
    "mn": ["manipur"],
    "ml": ["meghalaya"],
    "mz": ["mizoram"],
    "nl": ["nagaland"],
    "od": ["odisha", "orissa", "or"],
    "py": ["puducherry", "pondicherry"],
    "pb": ["punjab"],
    "rj": ["rajasthan"],
    "sk": ["sikkim"],
    "tn": ["tamil nadu", "tamilnadu"],
    "ts": ["telangana", "tg"],
    "tr": ["tripura"],
    "up": ["uttar pradesh"],
    "uk": ["uttarakhand", "uttaranchal", "ua"],
    "wb": ["west bengal"],
}

# region code -> region name + its departments. Sources disagree on which level
# they write ("Gironde" vs "Nouvelle-Aquitaine"), so departments map to regions.
FRANCE_REGIONS = {
    "ara": ["auvergne rhone alpes", "ain", "allier", "ardeche", "cantal", "drome",
            "isere", "loire", "haute loire", "puy de dome", "rhone", "savoie",
            "haute savoie"],
    "bfc": ["bourgogne franche comte", "cote d or", "doubs", "jura", "nievre",
            "haute saone", "saone et loire", "yonne", "territoire de belfort"],
    "bre": ["bretagne", "cotes d armor", "finistere", "ille et vilaine", "morbihan"],
    "cvl": ["centre val de loire", "cher", "eure et loir", "indre", "indre et loire",
            "loir et cher", "loiret"],
    "cor": ["corse", "corse du sud", "haute corse"],
    "ges": ["grand est", "ardennes", "aube", "marne", "haute marne",
            "meurthe et moselle", "meuse", "moselle", "bas rhin", "haut rhin", "vosges"],
    "hdf": ["hauts de france", "aisne", "nord", "oise", "pas de calais", "somme"],
    "idf": ["ile de france", "paris", "seine et marne", "yvelines", "essonne",
            "hauts de seine", "seine saint denis", "val de marne", "val d oise"],
    "nor": ["normandie", "calvados", "eure", "manche", "orne", "seine maritime"],
    "naq": ["nouvelle aquitaine", "charente", "charente maritime", "correze", "creuse",
            "dordogne", "gironde", "landes", "lot et garonne", "pyrenees atlantiques",
            "deux sevres", "vienne", "haute vienne"],
    "occ": ["occitanie", "ariege", "aude", "aveyron", "gard", "haute garonne", "gers",
            "herault", "lot", "lozere", "hautes pyrenees", "pyrenees orientales",
            "tarn", "tarn et garonne"],
    "pdl": ["pays de la loire", "loire atlantique", "maine et loire", "mayenne",
            "sarthe", "vendee"],
    "pac": ["provence alpes cote d azur", "paca", "alpes de haute provence",
            "hautes alpes", "alpes maritimes", "bouches du rhone", "var", "vaucluse"],
}


def _build_state_lookup():
    """country -> {folded surface form: canonical state code}."""
    us = {}
    for code, name in US_STATES.items():
        us[code] = us[name] = f"us_{code}"
    india = {}
    for code, names in INDIA_STATES.items():
        india[code] = f"in_{code}"
        for n in names:
            india[n] = f"in_{code}"
    fr = {}
    for code, names in FRANCE_REGIONS.items():
        for n in names:
            fr[n] = f"fr_{code}"
    return {"US": us, "India": india, "France": fr}


STATE_LOOKUP = _build_state_lookup()
STATE_LOOKUP_ANY = {k: v for table in STATE_LOOKUP.values() for k, v in table.items() if len(k) > 3}

# --------------------------------------------------------------------------
# Addresses: vocabulary -> canonical short form
# --------------------------------------------------------------------------

ADDR_ABBREV_COMMON = {
    "street": "st", "str": "st", "saint": "st",
    "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave",
    "drive": "dr", "driv": "dr", "lane": "ln", "court": "ct",
    "place": "pl", "circle": "cir", "boulevard": "blvd", "blv": "blvd",
    "highway": "hwy", "trail": "trl", "parkway": "pkwy", "terrace": "ter",
    "square": "sq", "building": "bldg", "floor": "fl", "apartment": "apt",
    "suite": "ste", "number": "no", "num": "no",
    "north": "n", "south": "s", "east": "e", "west": "w",
}
ADDR_ABBREV = {
    "US": {
        "expressway": "expy", "freeway": "fwy", "mount": "mt", "mountain": "mtn",
        "point": "pt", "heights": "hts", "junction": "jct", "center": "ctr",
        "centre": "ctr", "crossing": "xing",
        "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    },
    "India": {
        "nr": "near", "opp": "opposite", "sec": "sector", "sect": "sector",
        "gr": "ground", "grd": "ground", "flr": "fl", "flor": "fl",
        "col": "colony", "ngr": "nagar", "extn": "extension", "ext": "extension",
        "industrial": "indl", "ind": "indl",
    },
    "France": {
        "r": "rue", "bd": "blvd", "bld": "blvd", "all": "allee",
        "ch": "chemin", "chem": "chemin", "imp": "impasse", "rte": "route",
        "fg": "faubourg", "faub": "faubourg", "qu": "quai", "crs": "cours",
        "res": "residence", "resid": "residence", "ste": "st", "sainte": "st",
        "etage": "fl", "eme": "", "er": "", "bis": "", "ter": "",
    },
}

# Locality noise: "City of Menomonie", "Anoka CITY", "Bothel CDP", "Town Of Marion"
LOCALITY_NOISE = {"city", "of", "cdp", "town", "township", "village", "borough"}

# Address tokens that should not count as identifying words.
ADDR_STOPWORDS = {"no", "near", "opposite", "behind", "the", "and", "po", "box", "h",
                  "de", "la", "du", "des", "le", "les", "d", "l"}
