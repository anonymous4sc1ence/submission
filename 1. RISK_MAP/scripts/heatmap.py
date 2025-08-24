#!/usr/bin/env python3
r"""
heatmap.py – Visualize effective coverage for the N riskiest attacks (by weighted residual),
with attack codes in labels.

Usage (from RISK_MAP):  python .\scripts\heatmap.py
"""

from __future__ import annotations
import argparse, pathlib, re
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

def project_root() -> pathlib.Path: return pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = project_root()/ "data"
FIG_DIR  = project_root()/ "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)

def preferred_impl() -> pathlib.Path:
    # Prefer Digit, then Pepper, then G1_EDU, else first match
    order = ["Digit_implementation_status.csv", "Pepper_implementation_status.csv", "G1_EDU_implementation_status.csv"]
    for name in order:
        p = DATA_DIR / name
        if p.exists(): return p
    matches = sorted(DATA_DIR.glob("*_implementation_status.csv"))
    if not matches: raise FileNotFoundError("No *_implementation_status.csv in data/")
    return matches[0]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix",  default=str(DATA_DIR/"attacks_vs_defenses_normalised.csv"))
    p.add_argument("--weights", default=str(DATA_DIR/"attack_weights.csv"))
    p.add_argument("--impl",    default=str(preferred_impl()))
    p.add_argument("--code-map",default=str(DATA_DIR/"attack_code_map.csv"))
    p.add_argument("--top", type=int, default=10)
    return p.parse_args()

# code helpers
_DASHES = {"\u2010","\u2013","\u2014","\u2212"}
def _norm_sep(s:str)->str:
    for d in _DASHES: s=s.replace(d,"-")
    return s.replace("_","-")
def _extract_code_from_string(s:str)->Optional[str]:
    s=_norm_sep(s); m=re.match(r"\s*([A-Z]{1,3}-A\d{1,3})\b", s)
    if m: return m.group(1)
    m=re.match(r"([A-Z0-9]{2,5})\s+", s)
    return m.group(1) if (m and not m.group(1).isdigit()) else None
def _attack_labels(index: pd.Index, code_series: Optional[pd.Series])->List[str]:
    labels=[]; idx=set(code_series.index) if code_series is not None else set()
    for name in index:
        code = (str(code_series[name]) if code_series is not None and name in idx
                else _extract_code_from_string(name))
        if code:
            cn=_norm_sep(code); cleaned=re.sub(rf"^{re.escape(cn)}[\s:_\-]*","",_norm_sep(name)).strip()
            labels.append(f"{cn} {cleaned}")
        else: labels.append(name)
    return labels

def load(matrix, weights, impl, code_map):
    A = pd.read_csv(matrix, index_col="Attack Vector")
    code_series=None
    if "Code" in A.columns: code_series=A.pop("Code")
    elif pathlib.Path(code_map).exists():
        m = pd.read_csv(code_map)
        if {"Attack Vector","Code"}.issubset(m.columns):
            code_series = m.set_index("Attack Vector")["Code"]
    A = A.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    I = pd.read_csv(impl, index_col="Defence")["Implementation"].pipe(pd.to_numeric, errors="coerce").fillna(0).clip(0,1)
    W = pd.read_csv(weights, index_col="Attack Vector")["Weight"].pipe(pd.to_numeric, errors="coerce").fillna(0)
    common = A.columns.intersection(I.index)
    if common.empty: raise ValueError("No overlapping defences.")
    A, I = A[common], I[common]
    W = W.reindex(A.index).fillna(0)
    return A,I,W,code_series

def compute_topN(A,I,W,N):
    E = A.mul(I,axis=1); C = 1 - (1 - E).prod(axis=1)
    residual = W * (1 - C)
    N = max(1, min(int(N), len(residual)))
    idx = residual.sort_values(ascending=False).head(N).index
    return E.loc[idx]

def plot_heatmap(heat: pd.DataFrame, labels: List[str], out_png: pathlib.Path):
    fig, ax = plt.subplots(figsize=(13,5))
    im = ax.imshow(heat.values, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([c.split()[0] for c in heat.columns], rotation=90, fontsize=6)
    ax.set_title("Effective Coverage $E_{ij}$ – Top Weighted-Risk Attacks", pad=14, fontsize=12, weight="bold")
    cb = plt.colorbar(im, ax=ax, shrink=0.6); cb.set_label("Coverage (0..1)", rotation=270, labelpad=15)
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[✓] Heat-map → {out_png}")

def main():
    a = parse_args()
    A,I,W,code_series = load(a.matrix, a.weights, a.impl, a.code_map)
    heat = compute_topN(A,I,W,a.top); labels = _attack_labels(heat.index, code_series)
    robot = pathlib.Path(a.impl).stem.replace("_implementation_status","")
    out_png = FIG_DIR / robot / f"heatmap_top{len(heat)}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plot_heatmap(heat, labels, out_png)

if __name__=="__main__": main()
