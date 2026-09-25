import tomllib
from pathlib import Path
from types import SimpleNamespace

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


def _to_namespace(d):
    return SimpleNamespace(**{
        k: _to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()
    })


with open(CONFIG_PATH, "rb") as f:
    cfg = _to_namespace(tomllib.load(f))

# Module-level names used by main.py
_train = Path(cfg.datasetPaths.trainPath).expanduser()
_test = Path(cfg.datasetPaths.testPath).expanduser()

OUTPUT_DIR = Path(cfg.outputPaths.outputDir).expanduser()
TRAIN_SOURCE1_PATH = _train / "train_source1.tsv"
TRAIN_SOURCE2_PATH = _train / "train_source2.tsv"
TRAIN_SOURCE3_PATH = _train / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = Path(cfg.datasetPaths.gtPath).expanduser()
TEST_SOURCE1_PATH = _test / "test_source1.tsv"
TEST_SOURCE2_PATH = _test / "test_source2.tsv"
TEST_SOURCE3_PATH = _test / "test_source3.tsv"

MAX_CANDIDATES_PER_S1 = cfg.params.maxCandidatesPerS1
N_FOLDS = cfg.params.nFolds
