import sys
sys.path.insert(0, 'code/business_entity_resolution')
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# Test OCR normalization
def ocr_canonical(series: pd.Series) -> pd.Series:
    return (
        series.str.replace("1", "l", regex=False)
        .str.replace("0", "o", regex=False)
        .str.replace("vv", "w", regex=False)
        .str.replace("rn", "m", regex=False)
    )

s = pd.Series(["urban buslness", "urban business", "ex1 and brothers", "exl and brothers"])
print("Original:", s.tolist())
print("OCR canonical:", ocr_canonical(s).tolist())
res = process.cpdist(ocr_canonical(s[:2]), ocr_canonical(s[1:3]), scorer=fuzz.ratio, workers=-1)
print("OCR similarity:", res)

# Test number overlap
def num_overlap(nums1: pd.Series, nums2: pd.Series) -> np.ndarray:
    out = np.zeros(len(nums1), dtype=np.float32)
    s1_sets = [set(x.split()) if x else set() for x in nums1]
    s2_sets = [set(x.split()) if x else set() for x in nums2]
    for i, (a, b) in enumerate(zip(s1_sets, s2_sets)):
        if a and b:
            out[i] = 1.0 if (a & b) else -1.0
        else:
            out[i] = 0.0
    return out

n1 = pd.Series(["8 199 2 3", "10", "4303", ""])
n2 = pd.Series(["8 199 2", "7 10", "4304", "10"])
print("Num overlap:", num_overlap(n1, n2))
