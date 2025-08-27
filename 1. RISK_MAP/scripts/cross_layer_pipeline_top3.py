#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-layer pipeline → Top-3 Cross-Layer Cascades per Robot (2 hops)

Run:
  python scripts/cross_layer_pipeline_top3.py

Inputs (data/):
  - layer_dependency.csv              # optional; if present we reuse it
  - layer_edges_sem.csv               # required only if layer_dependency.csv missing (from,to,S,E,M)
  - robot_layer_coverage.csv          # Layer + robot columns (0..1 or 0..5)
  - attack_families_by_start.csv      # start_layer,attack_id,weight

Outputs:
  - data/S_matrix.csv, E_matrix.csv, M_matrix.csv       (if D built from S/E/M)
  - data/layer_dependency.csv                           (if built)
  - outputs_top3/possible_chains.csv
  - outputs_top3/attack_chains_auto.csv
  - outputs_top3/cascade_index.csv
  - outputs_top3/top3_by_robot.csv
  - outputs_top3/fig_top3_2hop_square.pdf/.png
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

# Path enumeration
EPS        = 0.05    # include edges with D_ij > EPS
MIN_PROP   = 0.04    # min product of D along a path
MAXLEN     = 6       # max nodes (we’ll filter exactly 2 hops later)
MONOTONIC  = True    # only allow forward layer order

# Figure
PALETTE    = {"Digit":"#4477AA","G1_EDU":"#DD8452","Pepper":"#55A868"}
RANK_STYLE = {0:"-",1:"--",2:":"}
FIGSIZE    = (10,10)
# ---------------------------------------

THIS   = Path(__file__).resolve()
ROOT   = THIS.parent.parent
DATA   = ROOT / "data"
OUTDIR = ROOT / "outputs_top3"
DATA.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
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

# ---------- enumerate chains ----------
IDX = {k:i for i,k in enumerate(LAYER_KEYS)}
def enumerate_paths(D: pd.DataFrame, eps: float, maxlen: int, min_prop: float, monotonic: bool) -> pd.DataFrame:
    adj = {u: [] for u in LAYER_KEYS}
    for i in LAYER_KEYS:
        for j in LAYER_KEYS:
            if i != j and float(D.loc[i,j]) > eps and ((not monotonic) or (IDX[i] < IDX[j])):
                adj[i].append(j)
    out = []
    def dfs(path, prod):
        u = path[-1]
        if len(path) >= 2 and prod >= min_prop:
            out.append((">".join(path), len(path)-1, prod))
        if len(path) == maxlen: return
        for v in adj[u]:
            if v in path:  # simple paths only
                continue
            dfs(path+[v], prod*float(D.loc[u,v]))
    for s in LAYER_KEYS:
        dfs([s], 1.0)
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
    return out[["chain_id","attack_id","weight","path"]]

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

def score_chains(D: pd.DataFrame, C: pd.DataFrame, chains: pd.DataFrame) -> pd.DataFrame:
    rows = []
    robots = list(C.columns)
    for _, r in chains.iterrows():
        w = float(r["weight"])
        nodes = _nodes(r["path"])
        prop = path_prop_factor(D, nodes)
        for robot in robots:
            Crow = C[robot]
            U = path_uncovered_product(Crow, nodes)
            m = path_min_coverage(Crow, nodes)
            CRR = float(np.clip(w*prop*U, 0.0, 1.0))
            CCI = float(np.clip(1.0-CRR, 0.0, 1.0))
            rows.append({
                "robot": robot,
                "attack_id": r["attack_id"],
                "path": ">".join(nodes),
                "weight": round(w,6),
                "prop_factor": round(prop,6),
                "uncovered_prod": round(U,6),
                "min_coverage": round(m,6),
                "CRR": round(CRR,6),
                "CCI": round(CCI,6),
            })
    return (pd.DataFrame(rows)
            .sort_values(["robot","CRR"], ascending=[True, False])
            .reset_index(drop=True))

# ---------- figure (Top-3 2-hop, de-duplicated by path) ----------
def draw_top3_square(scored: pd.DataFrame, outbase: Path):
    # Keep EXACTLY 2 hops
    scored = scored.copy()
    scored["hops"] = scored["path"].astype(str).str.count(">")
    scored = scored[scored["hops"] == 2]
    if scored.empty:
        raise SystemExit("[ERROR] No 2-hop chains after scoring.")

    # compact code and de-duplicate by robot+path
    scored["code"] = scored["attack_id"].apply(_attack_code)
    srt = scored.sort_values("CRR", ascending=False)
    agg = (srt.groupby(["robot","path"], as_index=False)
              .agg({"CRR":"max",
                    "prop_factor":"first",
                    "min_coverage":"first",
                    "attack_id":"first",
                    "code": lambda x: list(dict.fromkeys(x))}))
    def compact_codes(lst):
        if not lst: return ""
        primary, extras = lst[0], lst[1:]
        if not extras: return primary
        if len(extras)<=2: return primary + ", " + ", ".join(extras)
        return f"{primary} +{len(extras)}"
    agg["label_codes"] = agg["code"].apply(compact_codes)

    robots = sorted(agg["robot"].unique())
    for i, r in enumerate(robots):
        PALETTE.setdefault(r, f"C{i}")

    top3 = {r: agg[agg["robot"]==r].sort_values("CRR", ascending=False).head(3).copy()
            for r in robots}

    # save table too
    pd.concat(list(top3.values()), ignore_index=True).to_csv(OUTDIR/"top3_by_robot.csv", index=False)

    vmax = max((g["CRR"].max() for g in top3.values() if not g.empty), default=0.01)
    def lw(v): return 2.0 + 8.0 * (float(v) / max(vmax, 1e-6))

    x_pos = {LAYER_KEYS[i]: i for i in range(len(LAYER_KEYS))}
    row_y = {r: i + 0.5 for i, r in enumerate(robots)}
    offsets = [-0.18, 0.0, +0.18]

    fig = plt.figure(figsize=FIGSIZE)
    ax  = fig.add_subplot(111)
    ax.set_xlim(-0.6, len(LAYER_KEYS)-0.2)
    ax.set_ylim(0, max(row_y.values()) + 0.6)
    ax.axis("off")

    ymax = max(row_y.values()) + 0.4
    for i, layer in enumerate(LAYER_KEYS):
        ax.plot([i,i],[0,ymax], color="#E6E6E6", lw=1.0, zorder=0)
        ax.text(i, -0.12, layer, ha="center", va="top", fontsize=12)

    for r in robots:
        ax.text(-0.55, row_y[r], r, ha="right", va="center", fontsize=12, color=PALETTE[r])

    for r in robots:
        g = top3[r]
        for rank, row in enumerate(g.itertuples(index=False)):
            nodes = _nodes(row.path)
            if len(nodes) != 3:  # safety
                continue
            y  = row_y[r] + offsets[min(rank,2)]
            xs = [x_pos[n] for n in nodes]
            ax.plot(xs, [y]*len(xs),
                    linestyle=RANK_STYLE.get(rank,"-"),
                    color=PALETTE[r],
                    linewidth=lw(row.CRR),
                    alpha=0.95,
                    solid_capstyle="round")
            ax.scatter(xs, [y]*len(xs), s=28, color="white",
                       edgecolor=PALETTE[r], linewidth=1.4, zorder=3)
            ax.text(xs[-1]+0.12, y,
                    f"{'>'.join(nodes)} ({row.label_codes})  CRR={row.CRR:.3f}",
                    ha="left", va="center", fontsize=10, color=PALETTE[r])

    # Legends (use keyword 'handles=' to avoid the Artist/labels ambiguity)
    robot_handles = [Line2D([0],[0], color=PALETTE[r], lw=6, label=r) for r in robots]
    rank_handles  = [Line2D([0],[0], color="#555555", lw=3,
                            linestyle=RANK_STYLE[i], label=f"Rank #{i+1}") for i in range(3)]

    leg1 = ax.legend(handles=robot_handles, loc="upper left", bbox_to_anchor=(0.0, 1.03),
                     ncol=max(1,len(robots)), frameon=False, fontsize=11, title="Robot")
    ax.add_artist(leg1)
    ax.legend(handles=rank_handles, loc="upper right", bbox_to_anchor=(1.0, 1.03),
              ncol=3, frameon=False, fontsize=11, title="Chain rank")

    fig.suptitle("Top-3 Cross-Layer Cascades per Robot (2 hops)", y=0.98, fontsize=14)
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

    # 1) Enumerate all paths from D
    paths = enumerate_paths(D, EPS, MAXLEN, MIN_PROP, MONOTONIC)
    (OUTDIR/"possible_chains.csv").write_text(paths.to_csv(index=False), encoding="utf-8")

    # 2) Keep exactly 2-hop chains
    paths2 = paths[paths["hops"] == 2].reset_index(drop=True)
    if paths2.empty:
        raise SystemExit("[ERROR] No 2-hop paths; relax EPS/MIN_PROP or check D.")

    # 3) Map to provisional attack chains
    mapped = map_chains_to_attacks(paths2, fam)
    mapped.to_csv(OUTDIR/"attack_chains_auto.csv", index=False)

    # 4) Score CRR/CCI for every robot
    scored = score_chains(D, C, mapped)
    scored.to_csv(OUTDIR/"cascade_index.csv", index=False)

    # 5) Draw Top-3 figure
    draw_top3_square(scored, OUTDIR/"fig_top3_2hop_square")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
