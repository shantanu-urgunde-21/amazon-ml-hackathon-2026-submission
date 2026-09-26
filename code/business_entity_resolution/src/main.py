"""
Business Entity Resolution pipeline (Amazon ML Challenge 2026).

    python src/main.py                      # all stages: data -> blocking -> model -> output
    python src/main.py --stage evaluate     # one stage (earlier stages are loaded from cache)

Stages (see src/pipeline.py):
    prepare     learn Indic lexicon, normalize all six source files
    candidates  FAISS similarity-index candidate generation for train and test
    train       XGBoost (GPU if available) on candidate pairs of fit-split entities
    evaluate    tune the decision gate, report macro F0.5 on untouched holdout entities
    predict     score the test set, write output/candidate_pairs.tsv + matching_results.tsv
"""

import argparse
import sys
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from src import pipeline  # noqa: E402

STAGES = {
    "prepare": pipeline.prepare,
    "candidates": lambda: (pipeline.all_candidates("train"), pipeline.all_candidates("test")),
    "train": pipeline.train,
    "evaluate": pipeline.evaluate,
    "predict": pipeline.predict_test,
}


def main():
    parser = argparse.ArgumentParser(description="Run the Business Entity Resolution pipeline")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    args = parser.parse_args()

    start = time.time()
    for name, fn in STAGES.items():
        if args.stage in ("all", name):
            pipeline.log(f"=== stage: {name} ===")
            fn()
    pipeline.log(f"done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
