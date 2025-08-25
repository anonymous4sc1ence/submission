# Humanoid Security SoK — Artifact Repository (Anonymous)

> **Anonymous Artifact for USENIX Security (SoK)**  
> This repository accompanies the paper:  
> *“SoK: Cybersecurity Assessment of the Humanoid Ecosystem”*  

It contains two main components:  
1. **RISK-MAP** — the core method, data, and code to reproduce all figures, tables, and scores in the paper.  
2. **Related Works** — survey notes, bibliographic material, and literature mappings supporting the SoK.

---

## Repository Layout

```
submission/
├── 1. RISK_MAP/ # Artifact implementation (code, data, figures, scripts)
└── 2. Related_works/ # Literature notes and references supporting the SoK
```


---

## 1. RISK-MAP (Artifact)

The `1. RISK_MAP/` folder contains:

- **Core data**: attack–defense matrices, severity weights, and per-robot implementation CSVs.  
- **Scripts**: to compute scores, generate radar/heatmaps, and run Monte Carlo sensitivity analyses.  
- **Notebook**: `RISK_MAP_Assessment.ipynb` — main entrypoint for reproduction.  
- **Figures**: generated plots corresponding to paper results (Figures 4a/4b, Table 3, sensitivity tables).  

📖 See the detailed [RISK-MAP README](./1.%20RISK_MAP/readme.md) for step-by-step instructions to reproduce results.

---

## 2. Related Works

The `2. Related_works/` folder contains:

- Curated literature survey spreadsheets and notes.  
- Paper classification tables by attack/defense and layer.  
- Reference material used in constructing the SoK taxonomy.  

This folder supports transparency of the **systematization process** (PRISMA-style review + coverage tables) but is not required to reproduce numerical RISK-MAP outputs.

---

## How to Reproduce the Results

1. Navigate to the RISK-MAP folder:
   ```bash
   cd "submission/1. RISK_MAP"

```
2. Follow the instructions in its README to:
- Run the notebook for full reproduction, or
- Use scripts/ for command-line reproduction.

3. Generated figures and CSVs will appear under figures/ and data/sensitivity/.


```

