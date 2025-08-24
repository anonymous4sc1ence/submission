#!/usr/bin/env python3
"""
applicable_attacks_generator.py – Create full attack applicability CSVs
(one per robot) assuming all vectors are applicable.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Load all attack vectors
matrix_path = DATA_DIR / "attacks_vs_defenses_normalised.csv"
A = pd.read_csv(matrix_path, index_col="Attack Vector")
attack_vectors = A.index.to_series().reset_index(drop=True)
df = pd.DataFrame({"Attack Vector": attack_vectors})

# Robots
robots = ["Digit", "G1_EDU", "Pepper"]
for robot in robots:
    out_path = DATA_DIR / f"{robot}_applicable_attacks.csv"
    df.to_csv(out_path, index=False)
    print(f"[✓] Saved {out_path}")
