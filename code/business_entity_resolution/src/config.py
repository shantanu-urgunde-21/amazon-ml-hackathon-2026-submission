"""
Configuration module for Amazon ML Challenge 2026 - Business Entity Resolution.
Centralizes paths, hyperparameters, and processing constants.
"""

import os
from pathlib import Path

# Resolve base directories
# This file is located at <ROOT>/code/business_entity_resolution/src/config.py
SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent
PROJECT_ROOT = PKG_DIR.parent.parent

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Dataset directory resolution
# 1. Check local project root: PROJECT_ROOT / "dataset"
# 2. Check student resource directory: PROJECT_ROOT.parent / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
# 3. Environment variable override DATASET_DIR
env_dataset_dir = os.environ.get("DATASET_DIR")
if env_dataset_dir and Path(env_dataset_dir).exists():
    DATASET_DIR = Path(env_dataset_dir).resolve()
elif (PROJECT_ROOT / "dataset").exists():
    DATASET_DIR = PROJECT_ROOT / "dataset"
elif (PROJECT_ROOT.parent / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset").exists():
    DATASET_DIR = PROJECT_ROOT.parent / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
else:
    # Default fallback
    DATASET_DIR = PROJECT_ROOT / "dataset"

# Training datasets
TRAIN_DIR = DATASET_DIR / "train"
TRAIN_SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

# Test datasets
TEST_DIR = DATASET_DIR / "test"
TEST_SOURCE1_PATH = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH = TEST_DIR / "test_source3.tsv"

# Validation Script Path
VALIDATE_SUBMISSION_SCRIPT = (
    PROJECT_ROOT.parent / "6ab10eb3b23ba_student_resource" / "student_resource" / "utils" / "validate_submission.py"
)

# Blocking & Candidate Generation Hyperparameters
MAX_CANDIDATES_PER_S1 = 60          # Upper bound on candidates per S1 entity
MAX_TOKEN_BUCKET_SIZE = 1500        # Inverted index frequency ceiling for common words
MIN_NAME_TOKEN_LEN = 3              # Ignore 1-2 char tokens for name blocking
NGRAM_SIZE = 3                      # Character n-gram size

# Modeling & Validation
RANDOM_SEED = 42
N_FOLDS = 5
BETA = 0.5                          # F_beta metric beta parameter

# Decision Gate Search Bounds
DEFAULT_TAU_SINGLETON = 0.60
DEFAULT_TAU_MATCH = 0.55
DEFAULT_DELTA_PROB = 0.15

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Dir:  {DATASET_DIR}")
    print(f"Train S1:     {TRAIN_SOURCE1_PATH} (exists: {TRAIN_SOURCE1_PATH.exists()})")
    print(f"Test S1:      {TEST_SOURCE1_PATH} (exists: {TEST_SOURCE1_PATH.exists()})")
    print(f"Outputs:      {OUTPUT_DIR}")
