#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-layer pipeline → Top-K Cross-Layer Cascades per Robot (Mixed hops)

Run:
  python scripts/cross_layer_pipeline_topk_mixed.py

Inputs (data/):
  - layer_dependency.csv              # optional; if present we reuse it
  - layer_edges_sem.csv               # required only if layer_dependency.csv missing (from,to,S,E,M)
  - robot_layer_coverage.csv          # Layer + robot columns (0..1 or 0..5)
  - attack_families_by_start.csv      # start_layer,attack_id,weight

Outputs:
  - data/S_matrix.csv, E_matrix.csv, M_matrix.csv       (if D built from S/E/M)
  - data/layer_dependency.csv                           (if built)
  - outputs_topk_mixed/possible_chains.csv
  - outputs_topk_mixed/attack_chains_auto.csv
  - outputs_topk_mixed/cascade_index.csv
  - outputs_topk_mixed/topk_by_robot_mixed.csv
  - outputs_topk_mixed/topk_by_robot_h2.csv (and h3/h4 if allowed)
  - outputs_topk_mixed/fig_topk_mixed_square.pdf/.png
"""
from pathlib import Path
import re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------- config ----------------
LAYER_KEYS = ["P","S","DP","MW","DM","AP","SI"]

# D = (alpha*S + beta*E) * (1 - M), diag = 1
ALPHA, BETA  = 0.6, 0.4
D_DIAG       = 1.0
ROUND_DEC    = 2

# Path enumeration & filtering
EPS         = 0.05    # include edges with D_ij > EPS
MIN_PROP    = 0.04    # min product of D along a path
MAX_HOPS    = 4       # maximum hops to enumerate (e.g., allow up to 4-hop)
MONOTONIC   = True    # only allow forward layer order (P→…→SI)

# Mixed-hop selection
HOPS_ALLOWED = [2,3,4]   # which hop counts to include in "mixed"
TOPK         = 3         # top-K per robot

# Figure
PALETTE    = {"Digit":"#4477AA","G1_EDU":"#DD8452","Pepper":"#55A868"}
RANK_STYLE = {0:"-",1:"--",2:":"}
FIGSIZE    = (10,10)
# ---------------------------------------

THIS   = Path(__file__).resolve()
ROOT   = THIS.parent.parent
DATA   = ROOT / "data"
OUTDIR = ROOT / "outputs_topk_mixed"
DATA.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

SCORING_MODE = "geom_mean"   # "product" | "geom_mean" | "bottleneck" | "len_penalty"
TAU_LEN = 0.0                 # used only if SCORING_MODE == "len_penalty" (e.g., 0.5)

# ---- helper: compute path stats once ----
def path_stats(D: pd.DataFrame, nodes: list):
    vals = [float(D.loc[nodes[i], nodes[i+1]]) for i in range(len(nodes)-1)]
    if not vals:
        return 1.0, 1.0, 1.0
    prod = float(np.clip(np.prod(vals), 0.0, 1.0))
    gm = float(np.clip(prod ** (1.0/len(vals)), 0.0, 1.0))
    bottleneck = float(np.clip(min(vals), 0.0, 1.0))
    return prod, gm, bottleneck

def R(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (ROOT / p)

def _nodes(path_str: str):
    return [p.strip() for p in str(path_str).split(">") if p.strip()]

def _attack_code(attack_id: str) -> str:
    s = str(attack_id)
    m = re.search(r"\b([A-Z]{1,3}-A\d+)\b", s)
    if m: return m.group(1)
    last = s.split()[-1]
    return last if "-" in last else (s[:18]+"…" if len(s)>18 else s)

# ---------- S/E/M → D (or reuse) ----------
def _edges_to_matrix(edges: pd.DataFrame, col: str) -> pd.DataFrame:
    m = pd.DataFrame(0.0, index=LAYER_KEYS, columns=LAYER_KEYS)
    for _, r in edges.iterrows():
        m.loc[r["from"], r["to"]] = float(r[col])
    return m

def _save_square_matrix(mat: pd.DataFrame, out_csv: Path):
    df = mat.loc[LAYER_KEYS, LAYER_KEYS].copy()
    df.insert(0, "from", df.index)
    df.columns = ["from"] + [f"to{k}" for k in LAYER_KEYS]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

def make_or_load_D() -> pd.DataFrame:
    dep = DATA / "layer_dependency.csv"
    if dep.exists():
        df = pd.read_csv(dep).set_index("from")
        df.columns = [c.replace("to","") for c in df.columns]
        df = df.loc[LAYER_KEYS, LAYER_KEYS].astype(float)
        print(f"[i] Reusing {dep}")
        return df
    edges_csv = DATA / "layer_edges_sem.csv"
    if not edges_csv.exists():
        raise SystemExit("[ERROR] Missing data/layer_dependency.csv and data/layer_edges_sem.csv")
    edges = pd.read_csv(edges_csv)
    need = {"from","to","S","E","M"}
    if not need.issubset(edges.columns):
        raise SystemExit(f"[ERROR] {edges_csv} must have columns {sorted(need)}")
    for c in ["S","E","M"]:
        edges[c] = pd.to_numeric(edges[c], errors="coerce").astype(float).clip(0.0, 1.0)
    badf = sorted(set(edges["from"]) - set(LAYER_KEYS))
    badt = sorted(set(edges["to"]) - set(LAYER_KEYS))
    if badf or badt:
        raise SystemExit(f"[ERROR] Unknown layers in edges: from={badf} to={badt}")

    S = _edges_to_matrix(edges, "S")
    E = _edges_to_matrix(edges, "E")
    M = _edges_to_matrix(edges, "M")

    _save_square_matrix(S, DATA/"S_matrix.csv")
    _save_square_matrix(E, DATA/"E_matrix.csv")
    _save_square_matrix(M, DATA/"M_matrix.csv")

    if not np.isclose(ALPHA+BETA, 1.0, atol=1e-9):
        raise SystemExit(f"[ERROR] alpha+beta must equal 1.0 (got {ALPHA+BETA})")
    D = (ALPHA*S + BETA*E) * (1.0 - M)
    for l in LAYER_KEYS:
        D.loc[l,l] = D_DIAG
    D = D.clip(0,1).round(ROUND_DEC)
    _save_square_matrix(D, dep)
    print(f"[OK] Wrote {dep}")
    return D

# ---------- coverage & families ----------
def load_coverage() -> pd.DataFrame:
    p = DATA / "robot_layer_coverage.csv"
    if not p.exists(): raise SystemExit(f"[ERROR] Missing {p}")
    df = pd.read_csv(p)
    if "Layer" not in df.columns: raise SystemExit("[ERROR] robot_layer_coverage.csv needs a 'Layer' column")
    df = df.set_index("Layer").loc[LAYER_KEYS]
    if df.to_numpy().max() > 1.0:
        df = df / df.to_numpy().max()
    return df.clip(0,1)

def load_families() -> pd.DataFrame:
    p = DATA / "attack_families_by_start.csv"
    if not p.exists():
        raise SystemExit(f"[ERROR] Missing {p} (need start_layer,attack_id,weight)")
    df = pd.read_csv(p)
    need = {"start_layer","attack_id","weight"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[ERROR] {p} must have columns {sorted(need)}")
    return df

# ---------- enumerate chains (up to MAX_HOPS) ----------
IDX = {k:i for i,k in enumerate(LAYER_KEYS)}
def enumerate_paths(D: pd.DataFrame, eps: float, max_hops: int, min_prop: float, monotonic: bool) -> pd.DataFrame:
    """Enumerate simple paths up to max_hops edges (max_hops+1 nodes).
       Record any prefix path whose product >= min_prop."""
    adj = {u: [] for u in LAYER_KEYS}
    for i in LAYER_KEYS:
        for j in LAYER_KEYS:
            if i != j and float(D.loc[i,j]) > eps and ((not monotonic) or (IDX[i] < IDX[j])):
                adj[i].append(j)
    out = []
    def dfs(path, prod, hops):
        # record when >=1 hop and meets min_prop
        if hops >= 1 and prod >= min_prop:
            out.append((">".join(path), hops, prod))
        if hops == max_hops:
            return
        u = path[-1]
        for v in adj[u]:
            if v in path:  # simple paths only
                continue
            dfs(path+[v], prod*float(D.loc[u,v]), hops+1)
    for s in LAYER_KEYS:
        dfs([s], 1.0, 0)
    return (pd.DataFrame(out, columns=["path","hops","prop_factor"])
              .drop_duplicates()
              .sort_values(["hops","path"])
              .reset_index(drop=True))

# ---------- mapping & scoring ----------
def map_chains_to_attacks(cand: pd.DataFrame, fam: pd.DataFrame) -> pd.DataFrame:
    c = cand.copy()
    c["start_layer"] = c["path"].str.split(">").str[0]
    out = c.merge(fam, on="start_layer", how="inner")
    if out.empty:
        raise SystemExit("[ERROR] mapping produced no rows—check 'start_layer' values")
    out["chain_id"] = out["path"].apply(lambda s: f"CH_{abs(hash(s))%10**8:08d}")
    return out[["chain_id","attack_id","weight","path","hops"]]

def path_prop_factor(D: pd.DataFrame, nodes: list) -> float:
    prod = 1.0
    for i in range(len(nodes)-1):
        prod *= float(D.loc[nodes[i], nodes[i+1]])
    return float(np.clip(prod, 0.0, 1.0))

def path_uncovered_product(C_row: pd.Series, nodes: list) -> float:
    vals = [(1.0 - float(C_row.loc[n])) for n in nodes]
    return float(np.clip(np.prod(vals) if vals else 1.0, 0.0, 1.0))

def path_min_coverage(C_row: pd.Series, nodes: list) -> float:
    vals = [float(C_row.loc[n]) for n in nodes]
    return float(np.min(vals)) if vals else 0.0

# ---- replace your score_chains() with this version ----
def score_chains(D: pd.DataFrame, C: pd.DataFrame, chains: pd.DataFrame) -> pd.DataFrame:
    rows = []
    robots = list(C.columns)
    for _, r in chains.iterrows():
        w = float(r["weight"])
        nodes = _nodes(r["path"])
        h = len(nodes) - 1
        prod, gm, bottleneck = path_stats(D, nodes)
        for robot in robots:
            Crow = C[robot]
            U = path_uncovered_product(Crow, nodes)   # ∏(1 - C_r(ℓ))
            m = path_min_coverage(Crow, nodes)        # for diagnostics

            # Base CRR as defined (kept for reporting/plots)
            CRR = float(np.clip(w * prod * U, 0.0, 1.0))
            CCI = float(np.clip(1.0 - CRR, 0.0, 1.0))

            # --- length-aware comparison metric for ranking only ---
            if SCORING_MODE == "product":
                rank_metric = CRR
            elif SCORING_MODE == "geom_mean":
                # replace prod with per-hop geometric mean in the CRR skeleton
                CRR_gm = float(np.clip(w * gm * U, 0.0, 1.0))
                rank_metric = CRR_gm
            elif SCORING_MODE == "bottleneck":
                CRR_bn = float(np.clip(w * bottleneck * U, 0.0, 1.0))
                rank_metric = CRR_bn
            elif SCORING_MODE == "len_penalty":
                rank_metric = CRR / (h ** max(TAU_LEN, 0.0) if h > 0 else 1.0)
            else:
                rank_metric = CRR  # fallback

            rows.append({
                "robot": robot,
                "attack_id": r["attack_id"],
                "path": ">".join(nodes),
                "hops": h,
                "weight": round(w,6),
                "prop_factor": round(prod,6),
                "prop_gmean": round(gm,6),
                "prop_bottleneck": round(bottleneck,6),
                "uncovered_prod": round(U,6),
                "min_coverage": round(m,6),
                "CRR": round(CRR,6),
                "CCI": round(CCI,6),
                "rank_metric": round(float(rank_metric), 6),
            })
    return (pd.DataFrame(rows)
            .sort_values(["robot","rank_metric"], ascending=[True, False])
            .reset_index(drop=True))


# ---------- figure (Top-K mixed hops, de-duplicated by path) ----------
def draw_topk_mixed(scored: pd.DataFrame, outbase: Path, topk: int):
    srt = scored.sort_values("rank_metric", ascending=False)
    agg = (srt.groupby(["robot","path"], as_index=False)
              .agg({"rank_metric":"max",
                    "CRR":"first",
                    "prop_factor":"first",
                    "prop_gmean":"first",
                    "prop_bottleneck":"first",
                    "min_coverage":"first",
                    "hops":"first",
                    "attack_id":"first"}))
    robots = sorted(agg["robot"].unique())
    topk_by_robot = {r: agg[agg["robot"]==r].sort_values("rank_metric", ascending=False).head(topk).copy()
                     for r in robots}
    pd.concat(list(topk_by_robot.values()), ignore_index=True).to_csv(OUTDIR/"topk_by_robot_mixed.csv", index=False)
    # Per-hop CSVs too (nice for appendix)
    for h in sorted(scored["hops"].unique()):
        per_h = agg[agg["hops"]==h]
        per_h_out = per_h.sort_values(["robot","CRR"], ascending=[True,False]).groupby("robot").head(topk)
        per_h_out.to_csv(OUTDIR/f"topk_by_robot_h{h}.csv", index=False)

    # Palette defaults
    for i, r in enumerate(robots):
        PALETTE.setdefault(r, f"C{i}")

    # Visual encode rank (#1 solid, #2 dashed, #3 dotted)
    ranked_rows = []
    for r in robots:
        g = topk_by_robot[r].copy()
        g["rank"] = range(len(g))
        ranked_rows.append(g)
    vis = pd.concat(ranked_rows, ignore_index=True) if ranked_rows else agg.head(0)

    # Setup canvas
    x_pos = {LAYER_KEYS[i]: i for i in range(len(LAYER_KEYS))}
    row_y = {r: i + 0.5 for i, r in enumerate(robots)}
    offsets = [-0.18, 0.0, +0.18]

    vmax = float(vis["CRR"].max()) if not vis.empty else 0.01
    def lw(v): return 2.0 + 8.0 * (float(v) / max(vmax, 1e-6))

    fig = plt.figure(figsize=FIGSIZE)
    ax  = fig.add_subplot(111)
    ax.set_xlim(-0.6, len(LAYER_KEYS)-0.2)
    ax.set_ylim(0, max(row_y.values()) + 0.6 if row_y else 1.0)
    ax.axis("off")

    ymax = max(row_y.values()) + 0.4 if row_y else 1.0
    for i, layer in enumerate(LAYER_KEYS):
        ax.plot([i,i],[0,ymax], color="#E6E6E6", lw=1.0, zorder=0)
        ax.text(i, -0.12, layer, ha="center", va="top", fontsize=12)

    for r in robots:
        ax.text(-0.55, row_y[r], r, ha="right", va="center", fontsize=12, color=PALETTE[r])

    # Draw mixed-hop polylines
    for _, row in vis.iterrows():
        nodes = _nodes(row.path)
        y  = row_y[row.robot] + offsets[min(int(row["rank"]), len(offsets)-1)]
        xs = [x_pos[n] for n in nodes]
        ax.plot(xs, [y]*len(xs),
                linestyle=RANK_STYLE.get(int(row["rank"]), "-"),
                color=PALETTE[row.robot],
                linewidth=lw(row.CRR),
                alpha=0.95,
                solid_capstyle="round")
        ax.scatter(xs, [y]*len(xs), s=28, color="white",
                   edgecolor=PALETTE[row.robot], linewidth=1.4, zorder=3)
        ax.text(xs[-1]+0.12, y,
                f"{'>'.join(nodes)}  (h={row.hops})  CRR={row.CRR:.3f}",
                ha="left", va="center", fontsize=10, color=PALETTE[row.robot])

    # Legends
    robot_handles = [Line2D([0],[0], color=PALETTE[r], lw=6, label=r) for r in robots]
    rank_handles  = [Line2D([0],[0], color="#555555", lw=3,
                            linestyle=RANK_STYLE[i], label=f"Rank #{i+1}") for i in range(min(TOPK,3))]

    if robot_handles:
        leg1 = ax.legend(handles=robot_handles, loc="upper left", bbox_to_anchor=(0.0, 1.03),
                         ncol=max(1,len(robots)), frameon=False, fontsize=11, title="Robot")
        ax.add_artist(leg1)
    if rank_handles:
        ax.legend(handles=rank_handles, loc="upper right", bbox_to_anchor=(1.0, 1.03),
                  ncol=min(3,TOPK), frameon=False, fontsize=11, title="Chain rank")

    fig.suptitle(f"Top-{TOPK} Cross-Layer Cascades per Robot (Mixed hops: {HOPS_ALLOWED})",
                 y=0.98, fontsize=14)
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure → {outbase.with_suffix('.pdf')} and .png")

# ---------- main orchestration ----------
def main():
    # 0) Load/build core inputs
    D = make_or_load_D()
    C = load_coverage()
    fam = load_families()

    # 1) Enumerate all paths up to MAX_HOPS
    paths = enumerate_paths(D, EPS, MAX_HOPS, MIN_PROP, MONOTONIC)
    (OUTDIR/"possible_chains.csv").write_text(paths.to_csv(index=False), encoding="utf-8")

    # 2) Keep only allowed hop counts for the "mixed" view
    paths_allowed = paths[paths["hops"].isin(HOPS_ALLOWED)].reset_index(drop=True)
    if paths_allowed.empty:
        raise SystemExit("[ERROR] No paths at allowed hop counts; relax EPS/MIN_PROP or check D.")

    # 3) Map to provisional attack chains
    mapped = map_chains_to_attacks(paths_allowed, fam)
    mapped.to_csv(OUTDIR/"attack_chains_auto.csv", index=False)

    # 4) Score CRR/CCI for every robot
    scored = score_chains(D, C, mapped)
    scored.to_csv(OUTDIR/"cascade_index.csv", index=False)

    # 5) Emit per-hop Top-K CSVs and mixed Top-K figure
    draw_topk_mixed(scored, OUTDIR/"fig_topk_mixed_square", TOPK)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
