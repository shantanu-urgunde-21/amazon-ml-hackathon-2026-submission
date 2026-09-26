import tomllib
from pathlib import Path
from types import SimpleNamespace

SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent
PROJECT_ROOT = PKG_DIR.parent.parent

CONFIG_PATH = PKG_DIR / "config.toml"
CONFIG_EXAMPLE_PATH = PKG_DIR / "config.example.toml"


def _to_namespace(d):
    return SimpleNamespace(**{
        k: _to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()
    })


def _resolve(p):
    """Relative paths in config.toml are relative to PROJECT_ROOT, so the
    config works wherever the repo is cloned and whatever the cwd is."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


_cfg_file_to_read = CONFIG_PATH if CONFIG_PATH.exists() else CONFIG_EXAMPLE_PATH

with open(_cfg_file_to_read, "rb") as f:
    _raw = tomllib.load(f)

# Fallback dataset resolution if relative paths in config don't exist
_default_candidate_dataset_dirs = [
    PROJECT_ROOT / "student_resource" / "dataset",
    PROJECT_ROOT.parent / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset",
    PROJECT_ROOT / "dataset",
]
_resolved_dataset_dir = None
for _cand in _default_candidate_dataset_dirs:
    if _cand.exists():
        _resolved_dataset_dir = _cand
        break

for _k, _v in _raw["datasetPaths"].items():
    _res = _resolve(_v)
    if not _res.exists() and _resolved_dataset_dir:
        # Check if file/folder exists under detected dataset directory
        _sub_p = _resolved_dataset_dir / Path(_v).name
        if _sub_p.exists():
            _res = _sub_p
        elif "train" in _v and (_resolved_dataset_dir / "train").exists():
            _res = _resolved_dataset_dir / "train"
        elif "test" in _v and (_resolved_dataset_dir / "test").exists():
            _res = _resolved_dataset_dir / "test"
    _raw["datasetPaths"][_k] = str(_res)

_raw["outputPaths"]["outputDir"] = str(_resolve(_raw["outputPaths"]["outputDir"]))

cfg = _to_namespace(_raw)

# Module-level names (kept for main.py and the other pipeline modules)

# Output
OUTPUT_DIR = Path(cfg.outputPaths.outputDir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATCHING_RESULTS_PATH = OUTPUT_DIR / cfg.outputPaths.matchingResultsFile
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / cfg.outputPaths.candidatePairsFile

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

# Modeling & Validation
RANDOM_SEED = cfg.params.randomSeed
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
