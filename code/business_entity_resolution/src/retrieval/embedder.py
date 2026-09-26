"""Dense vectors for FAISS: char n-gram TF-IDF compressed with truncated SVD.

Why char n-grams: the noise in this data is typos, abbreviations, missing or
reordered words and transliteration residue. Character 2-4-grams inside word
boundaries survive all of these, and the SVD keeps the n-gram co-occurrence
structure in a small dense vector that FAISS can search.

Two embedders are fitted per country (name and address). A record's search
vector is their weighted concatenation, so cosine(search vectors) =
w * cos(name) + (1 - w) * cos(address). A record with an empty address gets a
zero address part and is searched on its name only.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize


def name_text(df: pd.DataFrame) -> pd.Series:
    return (df["name_core"] + " " + df["name_alias"]).str.strip()


def addr_text(df: pd.DataFrame) -> pd.Series:
    return (df["addr_norm"] + " " + df["state"]).str.strip()


class CharNgramEmbedder:
    def __init__(self, dim: int = 128, ngram_range=(2, 4), n_features: int = 2**18, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.hasher = HashingVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, n_features=n_features,
            alternate_sign=False, norm=None, dtype=np.float32,
        )
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.svd = TruncatedSVD(dim, algorithm="randomized", n_iter=4, random_state=seed)

    def fit(self, texts: pd.Series, sample: int = 200_000):
        rng = np.random.default_rng(self.seed)
        if len(texts) > sample:
            texts = texts.iloc[rng.choice(len(texts), sample, replace=False)]
        X = self.hasher.transform(texts)
        self.tfidf.fit(X)
        self.svd.fit(normalize(self.tfidf.transform(X)))
        self.svd.components_ = self.svd.components_.astype(np.float32)
        return self

    def transform(self, texts: pd.Series, chunk: int = 200_000, n_jobs: int = 6) -> np.ndarray:
        """n-gram hashing is pure Python, so it runs in worker processes; TF-IDF
        and the SVD projection are sparse matrix products done here. n_jobs is
        capped because every worker process holds its own copy of sklearn."""
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        starts = list(range(0, len(texts), chunk))
        hashed = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(self.hasher.transform)(texts.iloc[i:i + chunk].tolist()) for i in starts
        )
        for i, X in zip(starts, hashed):
            X = normalize(self.tfidf.transform(X))
            out[i:i + X.shape[0]] = normalize(self.svd.transform(X))
        return out


def search_vectors(name_vec: np.ndarray, addr_vec: np.ndarray, name_weight: float) -> np.ndarray:
    v = np.hstack([name_vec * np.sqrt(name_weight), addr_vec * np.sqrt(1.0 - name_weight)])
    return normalize(v).astype(np.float32)
