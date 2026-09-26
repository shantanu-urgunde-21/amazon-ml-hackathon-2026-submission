import tomllib
from pathlib import Path
from types import SimpleNamespace

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


def _to_namespace(d):
    return SimpleNamespace(**{
        k: _to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()
    })


# Base directories
# This file is located at <ROOT>/code/business_entity_resolution/src/config.py
SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent
PROJECT_ROOT = PKG_DIR.parent.parent


def _resolve(p):
    """Relative paths in config.toml are relative to PROJECT_ROOT, so the
    config works wherever the repo is cloned and whatever the cwd is."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


with open(CONFIG_PATH, "rb") as f:
    _raw = tomllib.load(f)

# Store resolved absolute paths back into cfg, so code reading cfg directly
# (e.g. notebooks running from src/notebooks/) gets usable paths too.
for _k, _v in _raw["datasetPaths"].items():
    _raw["datasetPaths"][_k] = str(_resolve(_v))
for _k in ("outputDir", "cacheDir"):
    _raw["outputPaths"][_k] = str(_resolve(_raw["outputPaths"][_k]))

cfg = _to_namespace(_raw)

# Module-level names (kept for main.py and the other pipeline modules)

# Output
OUTPUT_DIR = Path(cfg.outputPaths.outputDir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATCHING_RESULTS_PATH = OUTPUT_DIR / cfg.outputPaths.matchingResultsFile
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / cfg.outputPaths.candidatePairsFile
CACHE_DIR = Path(cfg.outputPaths.cacheDir)

# Training datasets
TRAIN_DIR = Path(cfg.datasetPaths.trainPath)
DATASET_DIR = TRAIN_DIR.parent
TRAIN_SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = Path(cfg.datasetPaths.gtPath)

# Test datasets
TEST_DIR = Path(cfg.datasetPaths.testPath)
TEST_SOURCE1_PATH = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH = TEST_DIR / "test_source3.tsv"

# Validation Script Path
VALIDATE_SUBMISSION_SCRIPT = Path(cfg.datasetPaths.validateScript)

# Blocking & Candidate Generation Hyperparameters
MAX_CANDIDATES_PER_S1 = cfg.params.maxCandidatesPerS1
MAX_TOKEN_BUCKET_SIZE = cfg.params.maxTokenBucketSize
MIN_NAME_TOKEN_LEN = cfg.params.minNameTokenLen
NGRAM_SIZE = cfg.params.ngramSize

# GPU (FAISS search + XGBoost), with automatic CPU fallback
USE_GPU = cfg.gpu.useGpu

# Phase 4: FAISS similarity index
EMBED_DIM = cfg.retrieval.embedDim
NAME_WEIGHT = cfg.retrieval.nameWeight
REV_K = cfg.retrieval.revK
REV_MARGIN = cfg.retrieval.revMargin
NPROBE = cfg.retrieval.nprobe
SEARCH_CHUNK = cfg.retrieval.searchChunk

# Modeling & Validation
TRAIN_ENTITIES = cfg.params.trainEntities
RANDOM_SEED = cfg.params.randomSeed
HOLDOUT_FRAC = cfg.params.holdoutFrac
N_FOLDS = cfg.params.nFolds
BETA = cfg.params.beta

# Decision Gate Search Bounds
DEFAULT_TAU_SINGLETON = cfg.params.defaultTauSingleton
DEFAULT_TAU_MATCH = cfg.params.defaultTauMatch
DEFAULT_DELTA_PROB = cfg.params.defaultDeltaProb

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Dir:  {DATASET_DIR}")
    print(f"Train S1:     {TRAIN_SOURCE1_PATH} (exists: {TRAIN_SOURCE1_PATH.exists()})")
    print(f"Test S1:      {TEST_SOURCE1_PATH} (exists: {TEST_SOURCE1_PATH.exists()})")
    print(f"Validator:    {VALIDATE_SUBMISSION_SCRIPT} (exists: {VALIDATE_SUBMISSION_SCRIPT.exists()})")
    print(f"Outputs:      {OUTPUT_DIR}")
