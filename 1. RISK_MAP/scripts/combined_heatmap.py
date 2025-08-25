#!/usr/bin/env python3
"""
score_RISK_MAP.py – Compute RISK_MAP scores:
- Radar + heatmap plots
- Combined radar plot
- Per-layer scores CSV
- Zero argument script
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────── Constants ─────────────────────────
LAYER_MAP = {
    'P': 'Physical', 'SP': 'Sensor and Perception', 'DP': 'Data Processing',
    'MW': 'Middleware', 'DM': 'Decision-Making',
    'AP': 'Application', 'SI': 'Social_Interface'
}
LAYER_ORDER = ['Physical', 'Sensor and Perception', 'Data Processing',
               'Middleware', 'Decision-Making', 'Application', 'Social_Interface']

# ───────────────────────── Paths ─────────────────────────
def project_root(): return pathlib.Path(__file__).resolve().parents[1]

ROOT = project_root()
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

MATRIX_PATH = DATA_DIR / "attacks_vs_defenses_normalised.csv"
WEIGHTS_PATH = DATA_DIR / "attack_weights.csv"
IMPL_FILES = sorted(DATA_DIR.glob("*_implementation_status.csv"))

# ───────────────────────── Core ─────────────────────────
def layer_of(defence_id: str) -> str:
    return LAYER_MAP.get(defence_id.split('-')[0], 'Other')

def compute_scores(A, W, I):
    common = A.columns.intersection(I.index)
    E = A[common].mul(I[common], axis=1)
    C = 1 - (1 - E).prod(axis=1)
    overall = (W * C).sum() / W.sum()

    layer_scores = {}
    for lyr in LAYER_ORDER:
        cols = [c for c in E.columns if layer_of(c) == lyr]
        if cols:
            C_lyr = 1 - (1 - E[cols]).prod(axis=1)
            layer_scores[lyr] = (W * C_lyr).sum() / W.sum()
        else:
            layer_scores[lyr] = 0.0
    return overall, layer_scores, E

# ───────────────────────── Plots ─────────────────────────
def radar_plot(layer_scores, overall_pct, out_pdf):
    labels = list(layer_scores.keys())
    radii = [v * 10 for v in layer_scores.values()]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    labels.append(labels[0]); radii.append(radii[0])
    angles = np.append(angles, angles[0])

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.plot(angles, radii, marker="o", linewidth=2.2, color="#1f77b4")
    ax.fill(angles, radii, alpha=0.25, color="#1f77b4")

    ax.set_ylim(0, 5)
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0.5, 5, 0.5), minor=True)
    ax.grid(True, which="major", color="grey", alpha=.6)
    ax.grid(True, which="minor", color="grey", linestyle=":", alpha=.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1])
    ax.set_title(f"RISK-MAP Layer Coverage (0–5)\nOverall: {overall_pct:.1f}%", pad=20)

    for ang, r in zip(angles[:-1], radii[:-1]):
        ax.text(ang, r + 0.15, f"{r:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

def combined_radar_plot(all_scores: dict[str, dict[str, float]], out_pdf: pathlib.Path):
    labels = LAYER_ORDER
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.append(angles, angles[0])

    colours = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
               "#9467bd", "#17becf", "#8c564b", "#e377c2"]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.set_ylim(0, 5)
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0.5, 5, 0.5), minor=True)
    ax.grid(True, which="major", color="grey", alpha=.6)
    ax.grid(True, which="minor", color="grey", linestyle=":", alpha=.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    for i, (robot, lyr_dict) in enumerate(all_scores.items()):
        radii = [v * 10 for v in lyr_dict.values()]
        radii.append(radii[0])
        c = colours[i % len(colours)]
        ax.plot(angles, radii, marker='o', linewidth=2, color=c, label=robot)
        ax.fill(angles, radii, alpha=0.15, color=c)
        for ang, r in zip(angles[:-1], radii[:-1]):
            ax.text(ang, r + 0.15, f"{r:.1f}", ha="center", va="bottom", fontsize=7, color=c)

    ax.set_title("RISK-MAP Layer Coverage – Combined View", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

def heatmap_top10(E, W, out_png):
    risk = W * (1 - E.sum(axis=1))
    idx = risk.nlargest(10).index
    data = E.loc[idx]
    if data.empty: return
    plt.figure(figsize=(10, 4))
    plt.imshow(data, aspect='auto', vmin=0, vmax=1)
    plt.yticks(range(len(idx)), idx, fontsize=7)
    plt.xticks(range(len(data.columns)), data.columns, rotation=90, fontsize=6)
    plt.colorbar(label='Coverage')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ───────────────────────── Main ─────────────────────────
def main():
    if not MATRIX_PATH.exists() or not WEIGHTS_PATH.exists():
        raise FileNotFoundError("Matrix or weights file not found.")
    if not IMPL_FILES:
        raise FileNotFoundError("No *_implementation_status.csv files found in data/")

    A = pd.read_csv(MATRIX_PATH, index_col='Attack Vector')
    W = pd.read_csv(WEIGHTS_PATH, index_col='Attack Vector')["Weight"]

    summary_rows = []
    all_layer_scores = {}

    for impl in IMPL_FILES:
        robot = impl.stem.replace("_implementation_status", "")
        I = pd.read_csv(impl, index_col='Defence')["Implementation"]

        overall, layer_scores, E = compute_scores(A, W, I)
        pct = overall * 100

        row = {"Robot": robot, "Overall": overall}
        row.update(layer_scores)
        summary_rows.append(row)
        all_layer_scores[robot] = layer_scores

        rdir = FIG_DIR / robot
        rdir.mkdir(parents=True, exist_ok=True)
        radar_plot(layer_scores, pct, rdir / "radar.pdf")
        heatmap_top10(E, W, rdir / "heatmap.png")

        print(f"[✓] {robot:>8}: RISK_MAP {pct:5.1f}%  → {rdir.relative_to(ROOT)}")

    # Save per-layer scores CSV
    df = pd.DataFrame(summary_rows).set_index("Robot")
    csv_out = DATA_DIR / "RISK_MAP_Per-Layer_Scores.csv"
    df.to_csv(csv_out)
    print(f"[✓] Saved per-layer scores CSV → {csv_out}")

    # Generate combined radar
 
    combined_out = FIG_DIR / "RISK_MAP_combined_radar.pdf"
    combined_radar_plot(all_layer_scores, combined_out)
    print(f"[✓] Combined radar saved → {combined_out}")

if __name__ == "__main__":
    main()
