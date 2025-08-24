#!/usr/bin/env python3
r"""
monte_carlo_RISK_MAP.py – Monte Carlo jitter of W and I; save run CSVs to data/sensitivity/,
and histograms to figures/sensitivity/.

Usage (from RISK_MAP):  python .\scripts\monte_carlo_RISK_MAP.py
"""

from __future__ import annotations
import argparse, pathlib
import numpy as np, pandas as pd

def project_root() -> pathlib.Path: return pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = project_root() / "data"
FIG_DIR  = project_root() / "figures"
OUT_DATA_DIR = DATA_DIR / "sensitivity"
OUT_FIG_DIR  = FIG_DIR  / "sensitivity"
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

def find_impl_files() -> list[pathlib.Path]:
    return sorted(DATA_DIR.glob("*_implementation_status.csv"))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix",  default=str(DATA_DIR/"attacks_vs_defenses_normalised.csv"))
    p.add_argument("--weights", default=str(DATA_DIR/"attack_weights.csv"))
    p.add_argument("--impl", nargs="+", default=None)
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--jitter", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hist", action="store_true")
    return p.parse_args()

def load(matrix, weights, impl):
    A = pd.read_csv(matrix,  index_col="Attack Vector")
    W = pd.read_csv(weights, index_col="Attack Vector")["Weight"]
    I = pd.read_csv(impl,    index_col="Defence")["Implementation"]
    common = A.columns.intersection(I.index)
    if common.empty: raise ValueError(f"No overlapping defences for {impl}")
    A = A[common].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    I = pd.to_numeric(I[common], errors="coerce").fillna(0.0).clip(0,1)
    W = pd.to_numeric(W, errors="coerce").fillna(0.0).reindex(A.index).fillna(0.0)
    return A,W,I

def score(A, W, I):
    E = A.mul(I, axis=1); C = 1 - (1 - E).prod(axis=1)
    denom = W.sum() if W.sum() else 1.0
    return float((W*C).sum()/denom)

def jitter_series(s: pd.Series, jitter: float, rng: np.random.Generator,
                  clamp01=False, nonneg=False):
    low, high = 1-jitter, 1+jitter
    out = s * rng.uniform(low, high, size=len(s))
    if clamp01: out = out.clip(0,1)
    if nonneg:  out = out.clip(lower=0)
    return out

def run_mc(A, W, I, n, jitter, rng) -> pd.Series:
    vals = np.empty(n, dtype=float)
    for k in range(n):
        Wj = jitter_series(W, jitter, rng, nonneg=True)
        Ij = jitter_series(I, jitter, rng, clamp01=True)
        vals[k] = score(A, Wj, Ij)
    return pd.Series(vals)

def maybe_hist(series: pd.Series, outpng: pathlib.Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(series.values, bins=40)
    ax.set_xlabel("RISK_MAP score"); ax.set_ylabel("Count")
    ax.set_title(f"Monte Carlo – {outpng.stem.replace('histogram_','')}")
    fig.tight_layout(); fig.savefig(outpng); plt.close(fig); return True

def main():
    a = parse_args()
    impl_files = [pathlib.Path(p) for p in (a.impl or find_impl_files())]
    if not impl_files:
        raise FileNotFoundError("No *_implementation_status.csv files found in data/.")

    base_ss = np.random.SeedSequence(a.seed)
    stats = []
    for impl in impl_files:
        name = impl.stem.replace("_implementation_status","")
        rng = np.random.default_rng(base_ss.spawn(1)[0])
        A,W,I = load(a.matrix, a.weights, impl)
        series = run_mc(A, W, I, a.runs, a.jitter, rng)
        (OUT_DATA_DIR / f"sensitivity_{name}.csv").write_text(series.to_csv(index=False), encoding="utf-8")
        if a.hist: maybe_hist(series, OUT_FIG_DIR / f"histogram_{name}.png")
        stats.append({"Robot":name, "Mean":float(series.mean()), "Std":float(series.std(ddof=1)),
                      "Min":float(series.min()), "Max":float(series.max())})
    pd.DataFrame(stats).to_csv(OUT_DATA_DIR / "table_sensitivity.csv", index=False)
    print(f"[✓] Wrote run CSVs & table to {OUT_DATA_DIR}")
    if a.hist: print(f"[✓] Histograms → {OUT_FIG_DIR}")

if __name__ == "__main__":
    main()
