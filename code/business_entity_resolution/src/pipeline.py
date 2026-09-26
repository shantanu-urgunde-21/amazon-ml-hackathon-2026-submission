"""Pipeline stages. Each stage caches its output under config.CACHE_DIR, so a
later stage can be re-run without redoing the earlier ones.

    prepare     lexicon (fit split only) + normalized parquet for all six files
    candidates  FAISS reverse-assignment candidates for every S1 entity of a split
    train       features for a sample of fit-split entities -> 5-fold XGBoost (GPU if available)
    evaluate    tune the gate on half of the holdout, report macro F0.5 on the other half
    predict     test candidates -> model -> exclusive assignment -> gate -> output TSVs

Leakage protocol (train/validation on the training set):
  * train S1 entities are split once into `fit` (80%) and `holdout` (20%), by
    entity. Every S2/S3 record belongs to at most one S1 entity, so no true
    pair is shared between the two sides.
  * label-driven steps use `fit` only: Indic lexicon, XGBoost (GroupKFold by
    S1 entity inside `fit`), retrieval settings (chosen on fit entities).
  * the gate thresholds are tuned on one half of `holdout`; the benchmark is
    reported on the other half, which nothing was fitted or tuned on.
  * exclusive assignment during evaluation uses out-of-fold probabilities for
    the trained entities, never in-sample ones.
  * embedders / FAISS index are unsupervised (no labels) and are fitted the
    same way on the test set at prediction time.
"""

import gc
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import src.config as config
from src.dataloader import build_lexicon, build_normalized, load_ground_truth, load_source, split_s1_ids
from src.dataloader.loader import lexicon_fingerprint
from src.features import FEATURE_NAMES, build_features
from src.filters import evaluate_blocking_recall, generate_candidates
from src.models import cross_validate_model, feature_importance, predict
from src.normalization import load_lexicon, save_lexicon
from src.prediction import exclusive_assignment, macro_fbeta_pairs, optimize_decision_thresholds, select_pairs

SOURCES = {
    "train": [config.TRAIN_SOURCE1_PATH, config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH],
    "test": [config.TEST_SOURCE1_PATH, config.TEST_SOURCE2_PATH, config.TEST_SOURCE3_PATH],
}
CACHE = config.CACHE_DIR
SPLIT_PATH = CACHE / "s1_split.json"
LEXICON_PATH = CACHE / "indic_lexicon.json"
MODEL_PATH = CACHE / "models.pkl"
OOF_PATH = CACHE / "oof_train.parquet"
GATE_PATH = CACHE / "gate.json"
REPORT_PATH = CACHE / "benchmark.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def retrieval_params() -> dict:
    return {
        "dim": config.EMBED_DIM, "name_weight": config.NAME_WEIGHT, "rev_k": config.REV_K,
        "rev_margin": config.REV_MARGIN, "nprobe": config.NPROBE, "chunk": config.SEARCH_CHUNK,
        "max_candidates": config.MAX_CANDIDATES_PER_S1, "seed": config.RANDOM_SEED,
    }


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def get_split():
    """Train S1 entities split into fit / holdout (deterministic, cached)."""
    if SPLIT_PATH.exists():
        return json.loads(SPLIT_PATH.read_text())
    gt = load_ground_truth(config.TRAIN_GROUND_TRUTH_PATH)
    fit, holdout = split_s1_ids(list(gt), config.HOLDOUT_FRAC, config.RANDOM_SEED)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps({"fit": fit, "holdout": holdout}))
    return {"fit": fit, "holdout": holdout}


def get_lexicon():
    """Indic -> English lexicon learned from the fit split's links only."""
    if LEXICON_PATH.exists():
        return load_lexicon(LEXICON_PATH)
    log("learning Indic lexicon from fit-split ground truth ...")
    gt = load_ground_truth(config.TRAIN_GROUND_TRUTH_PATH)
    s1 = load_source(config.TRAIN_SOURCE1_PATH)
    pool = pd.concat([load_source(p) for p in SOURCES["train"][1:]], ignore_index=True)
    lexicon = build_lexicon(s1, pool, gt, get_split()["fit"])
    save_lexicon(lexicon, LEXICON_PATH)
    log(f"  lexicon: {len(lexicon)} words -> {LEXICON_PATH}")
    return lexicon


def load_split(split: str, country: str = None, columns=None):
    """(S1 frame, pool frame = S2 + S3) normalized, for 'train' or 'test'.
    With `country`, only that country's files are read (and only `columns`)."""
    lexicon = get_lexicon()
    frames = []
    for p in SOURCES[split]:
        files = build_normalized(p, lexicon, CACHE / "normalized")
        wanted = [files[country]] if country in files else ([] if country else list(files.values()))
        frames.append(pd.concat([pd.read_parquet(f, columns=columns) for f in wanted], ignore_index=True)
                      if wanted else pd.DataFrame(columns=columns))
    return frames[0], pd.concat(frames[1:], ignore_index=True)


def countries(split: str):
    return sorted(build_normalized(SOURCES[split][0], get_lexicon(), CACHE / "normalized"))


def prepare():
    lexicon = get_lexicon()
    for paths in SOURCES.values():
        for p in paths:
            t = time.time()
            files = build_normalized(p, lexicon, CACHE / "normalized")
            log(f"  normalized {p.name}: {', '.join(files)} ({time.time() - t:.0f}s)")


# --------------------------------------------------------------------------
# candidates (per country: pool records only ever match S1 of their country)
# --------------------------------------------------------------------------

RETRIEVAL_COLUMNS = ["entity_id", "country", "name_core", "name_alias", "addr_norm", "state"]
FEATURE_COLUMNS = ["entity_id", "country", "name_core", "name_alias", "name_compact", "name_phon",
                   "legal", "addr_norm", "state", "house_num", "addr_nums", "has_indic"]


def _country_files(split, country):
    """Normalized parquet files of a country: S1 file, [S2 file, S3 file]."""
    lexicon = get_lexicon()
    files = [build_normalized(p, lexicon, CACHE / "normalized").get(country) for p in SOURCES[split]]
    return files[0], [f for f in files[1:] if f is not None]


def _pool_batches(files, columns, batch_rows):
    """(start_row, frame) batches over the pool files, in load_split's row order."""
    start = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=batch_rows, columns=columns):
            frame = batch.to_pandas()
            yield start, frame
            start += len(frame)


def candidates_key() -> str:
    """Cache key of candidate files: normalized-data version + retrieval settings."""
    key = json.dumps([lexicon_fingerprint(get_lexicon()), retrieval_params()], sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:10]


def candidates(split: str, country: str):
    """Candidate pairs (row positions within the country's S1 / pool frames)."""
    path = CACHE / "candidates" / f"{split}_{country}.{candidates_key()}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    s1_file, pool_files = _country_files(split, country)
    s1 = pd.read_parquet(s1_file, columns=RETRIEVAL_COLUMNS)
    n_pool = sum(pq.ParquetFile(f).metadata.num_rows for f in pool_files)
    log(f"candidates [{split}/{country}]: S1 {len(s1):,}, pool {n_pool:,}")
    rng = np.random.default_rng(config.RANDOM_SEED)
    fit = pd.concat([pd.read_parquet(f, columns=RETRIEVAL_COLUMNS) for f in pool_files], ignore_index=True)
    fit = fit.iloc[np.sort(rng.choice(len(fit), min(len(fit), 200_000), replace=False))].reset_index(drop=True)
    gc.collect()
    batches = _pool_batches(pool_files, RETRIEVAL_COLUMNS, config.SEARCH_CHUNK)
    cands = generate_candidates(s1, batches, fit, retrieval_params(), log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    cands.to_parquet(path, index=False)
    log(f"  {len(cands):,} pairs ({len(cands) / max(1, len(s1)):.2f} per S1)")
    return cands


def all_candidates(split: str):
    for c in countries(split):
        candidates(split, c)


# --------------------------------------------------------------------------
# train / evaluate
# --------------------------------------------------------------------------


def load_truth():
    """(owner, n_links): owner maps each linked S2/S3 id to its S1 id (every pool
    record has at most one); n_links counts true links per S1 id. Arrow-backed
    strings, far lighter than a dict of Python sets."""
    gt = pd.read_csv(config.TRAIN_GROUND_TRUTH_PATH, sep="\t", dtype="string[pyarrow]", keep_default_na=False)
    n_links = gt["matched_entity_ids"].str.count(",").add(1).where(gt["matched_entity_ids"] != "", 0)
    n_links.index = gt["source1_entity_id"]
    links = gt[gt["matched_entity_ids"] != ""].assign(m=lambda d: d["matched_entity_ids"].str.split(",")).explode("m")
    owner = pd.Series(links["source1_entity_id"].to_numpy(), index=links["m"].astype("string[pyarrow]").to_numpy())
    return owner, n_links.astype(np.int32)


def _pair_ids(cands, s1, pool):
    return s1["entity_id"].to_numpy()[cands["s1_idx"]], pool["entity_id"].to_numpy()[cands["pool_idx"]]


def _labels(s1_ids, cand_ids, owner):
    """1 where the candidate's true S1 entity is this S1 entity (unlinked records -> 0)."""
    return (owner.reindex(cand_ids).fillna("").to_numpy(dtype=object) == s1_ids).astype(np.int8)


def _entity_blocks(cands, chunk_entities):
    """Row masks of `cands`, one per block of S1 entities (group features need
    all candidates of an entity in the same block)."""
    s1_idx = cands["s1_idx"].to_numpy()
    ents = np.unique(s1_idx)
    for i in range(0, len(ents), chunk_entities):
        yield np.isin(s1_idx, ents[i:i + chunk_entities])


def _features_in_chunks(s1, pool, cands, chunk_entities=150_000):
    """Feature matrix for all rows of `cands` (use for training-size inputs)."""
    X = pd.DataFrame(np.zeros((len(cands), len(FEATURE_NAMES)), np.float32), columns=FEATURE_NAMES)
    for mask in _entity_blocks(cands, chunk_entities):
        X.iloc[np.flatnonzero(mask)] = build_features(s1, pool, cands[mask].reset_index(drop=True)).to_numpy()
    return X


def _predict_in_chunks(models, s1, pool, cands, chunk_entities=150_000):
    """Model probabilities for all rows of `cands`, never holding the full feature matrix."""
    prob = np.zeros(len(cands), np.float32)
    for mask in _entity_blocks(cands, chunk_entities):
        prob[mask] = predict(models, build_features(s1, pool, cands[mask].reset_index(drop=True)))
    return prob


def train():
    owner, _ = load_truth()
    fit_all = np.array(get_split()["fit"])
    rng = np.random.default_rng(config.RANDOM_SEED)
    fit_ids = set(fit_all[rng.choice(len(fit_all), min(config.TRAIN_ENTITIES, len(fit_all)), replace=False)])
    Xs, ys, groups, keys = [], [], [], []
    for country in countries("train"):
        cands = candidates("train", country)
        s1, pool = load_split("train", country, FEATURE_COLUMNS)
        sel = cands[s1["entity_id"].isin(fit_ids).to_numpy()[cands["s1_idx"]]].reset_index(drop=True)
        s1_ids, cand_ids = _pair_ids(sel, s1, pool)
        Xs.append(_features_in_chunks(s1, pool, sel))
        ys.append(_labels(s1_ids, cand_ids, owner))
        groups.append(s1_ids)
        keys.append(sel[["s1_idx", "pool_idx"]].assign(country=country))
        log(f"  train features [{country}]: {len(sel):,} pairs")
        del s1, pool, cands
    X = pd.concat(Xs, ignore_index=True)
    y = np.concatenate(ys)
    log(f"train: {len(X):,} pairs from {len(fit_ids):,} fit entities, positive rate {y.mean():.3f}")
    oof, models = cross_validate_model(X, y, np.concatenate(groups), n_splits=config.N_FOLDS)
    # Out-of-fold probabilities of the trained entities. evaluate() uses them
    # instead of in-sample predictions, so the exclusive assignment on the
    # holdout never competes against memorized (overconfident) training pairs.
    pd.concat(keys, ignore_index=True).assign(prob=oof).to_parquet(OOF_PATH, index=False)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"models": models, "features": FEATURE_NAMES}, f)
    imp = feature_importance(models, FEATURE_NAMES)
    log("  top features: " + ", ".join(f"{k}={v:.0f}" for k, v in imp.sort_values(ascending=False).head(10).items()))
    return models


def _load_models():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)["models"]


def _score_country(split, country, models):
    """(s1, pool, cands, prob) for one country; prob after exclusive assignment."""
    cands = candidates(split, country)
    s1, pool = load_split(split, country, FEATURE_COLUMNS)
    prob = _predict_in_chunks(models, s1, pool, cands)
    if split == "train" and OOF_PATH.exists():  # no in-sample predictions (see train())
        oof = pd.read_parquet(OOF_PATH, filters=[("country", "==", country)])
        at = pd.MultiIndex.from_frame(cands[["s1_idx", "pool_idx"]]).get_indexer(
            pd.MultiIndex.from_frame(oof[["s1_idx", "pool_idx"]]))
        prob[at[at >= 0]] = oof["prob"].to_numpy()[at >= 0]
        log(f"  replaced {int((at >= 0).sum()):,} in-sample predictions with out-of-fold ones")
    prob = exclusive_assignment(cands["s1_idx"].to_numpy(), cands["pool_idx"].to_numpy(), prob)
    log(f"  scored {len(cands):,} pairs [{split}/{country}]")
    return s1, pool, cands, prob


def evaluate():
    """Gate tuned on one half of the holdout, benchmark reported on the other."""
    models = _load_models()
    owner, n_links = load_truth()
    holdout = np.array(get_split()["holdout"])
    tune_mask = np.random.default_rng(config.RANDOM_SEED + 1).random(len(holdout)) < 0.5
    halves = {"tune": set(holdout[tune_mask]), "report": set(holdout[~tune_mask])}

    # per holdout entity: its pairs (prob, label) and number of true links
    rows = {h: {"ent": [], "pair_ent": [], "pair_cand": [], "prob": [], "label": [], "country": []} for h in halves}
    blocking = {"captured": 0, "total": 0, "pairs": 0, "entities": 0}
    for country in countries("train"):
        s1, pool, cands, prob = _score_country("train", country, models)
        s1_ids, cand_ids = _pair_ids(cands, s1, pool)
        for h, ids in halves.items():
            ents = [e for e in s1["entity_id"] if e in ids]
            m = np.fromiter((s in ids for s in s1_ids), bool, len(s1_ids))
            rows[h]["ent"] += ents
            rows[h]["country"] += [country] * len(ents)
            rows[h]["pair_ent"].append(s1_ids[m])
            rows[h]["pair_cand"].append(cand_ids[m])
            rows[h]["prob"].append(prob[m])
            lab = _labels(s1_ids[m], cand_ids[m], owner)
            rows[h]["label"].append(lab)
            if h == "report":
                blocking["captured"] += int(lab.sum())
                blocking["total"] += int(n_links.reindex(ents).sum())
                blocking["pairs"] += int(m.sum())
                blocking["entities"] += len(ents)
        del s1, pool, cands, prob

    def arrays(h, country=None):
        r = rows[h]
        ents = [e for e, c in zip(r["ent"], r["country"]) if country is None or c == country]
        pos = pd.Series(np.arange(len(ents)), index=pd.Index(ents))
        pe = np.concatenate(r["pair_ent"])
        keep = pos.index.get_indexer(pe) >= 0
        return (pos.index.get_indexer(pe[keep]), np.concatenate(r["prob"])[keep],
                np.concatenate(r["label"])[keep], n_links.reindex(ents).to_numpy())

    thresholds, tune_score = optimize_decision_thresholds(*arrays("tune"), log=log)
    g, p, lab, nt = arrays("report")
    report_score = macro_fbeta_pairs(g, select_pairs(g, p, **thresholds), lab, nt)
    by_country = {}
    for c in sorted(set(rows["report"]["country"])):
        g, p, lab, nt = arrays("report", c)
        by_country[c] = round(macro_fbeta_pairs(g, select_pairs(g, p, **thresholds), lab, nt), 4)
    report = {
        "holdout_report_entities": blocking["entities"],
        "macro_f05_report_half": round(report_score, 4),
        "macro_f05_tune_half": round(tune_score, 4),
        "macro_f05_by_country": by_country,
        "thresholds": thresholds,
        "blocking_recall": round(blocking["captured"] / max(1, blocking["total"]), 4),
        "avg_candidates_per_s1": round(blocking["pairs"] / max(1, blocking["entities"]), 2),
    }
    GATE_PATH.write_text(json.dumps(thresholds))
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    # report-half pairs with decisions, for error analysis
    r = rows["report"]
    pe = np.concatenate(r["pair_ent"])
    g_all = pd.Index(r["ent"]).get_indexer(pe)
    p_all = np.concatenate(r["prob"])
    pd.DataFrame({
        "source1_entity_id": pe, "entity_id": np.concatenate(r["pair_cand"]), "prob": p_all,
        "label": np.concatenate(r["label"]), "selected": select_pairs(g_all, p_all, **thresholds),
    }).to_parquet(CACHE / "holdout_report_pairs.parquet", index=False)
    log(f"BENCHMARK (untouched holdout half): {json.dumps(report)}")
    return report


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------


def write_submission(s1_ids, cand_map, match_map):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CANDIDATE_PAIRS_PATH, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s in s1_ids:
            f.write(f"{s}\t{','.join(cand_map.get(s, []))}\n")
    with open(config.MATCHING_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in s1_ids:
            allowed = set(cand_map.get(s, []))
            f.write(f"{s}\t{','.join(m for m in match_map.get(s, []) if m in allowed)}\n")
    log(f"  wrote {config.CANDIDATE_PAIRS_PATH} and {config.MATCHING_RESULTS_PATH}")


def predict_test():
    models = _load_models()
    thresholds = json.loads(GATE_PATH.read_text())
    cand_map, match_map, n_pairs, n_sel = {}, {}, 0, 0
    for country in countries("test"):
        s1, pool, cands, prob = _score_country("test", country, models)
        s1_ids, cand_ids = _pair_ids(cands, s1, pool)
        sel = select_pairs(cands["s1_idx"].to_numpy(), prob, **thresholds) if len(cands) else np.zeros(0, bool)
        cand_map.update(pd.Series(cand_ids).groupby(s1_ids).agg(list).to_dict())
        match_map.update(pd.Series(cand_ids[sel]).groupby(s1_ids[sel]).agg(list).to_dict())
        n_pairs, n_sel = n_pairs + len(cands), n_sel + int(sel.sum())
        del s1, pool, cands, prob
    s1_all = load_source(config.TEST_SOURCE1_PATH, usecols=["entity_id"])["entity_id"].tolist()
    write_submission(s1_all, cand_map, match_map)
    log(f"  test: {len(s1_all):,} S1 | candidates/S1 {n_pairs / len(s1_all):.2f} | "
        f"matched pairs {n_sel:,} | S1 with a match {len(match_map):,}")
