# RISK-MAP: A Comprehensive Framework for Robotics Security Assessment

[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxxx-blue)](https://doi.org/10.xxxx/xxxxxx)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


> **Artifact Submission for USENIX Security 2026**  
> This repository contains the artifact for the paper “SoK: Cybersecurity Assessment of Humanoid Ecosystem.”  
> Authors: [Anonymous for Review]
> The code and data that reproduce the paper’s figures and scores live under **`1. RISK_MAP/`**. The bibliography and supporting notes are under **`2. Related_works/`**.

## Abstract

RISK-MAP is a novel method for comprehensive security assessment of robotic systems across multiple architectural layers. This repository contains the complete implementation, datasets, and experimental artifacts supporting our USENIX Security 2025 submission.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Dataset Description](#dataset-description)
- [Extending the Framework](#extending-the-framework)
- [Citation](#citation)
- [License](#license)

## 🔍 Overview

The RISK-MAP framework provides:

- **Multi-layer Security Assessment**: Evaluates security across 7 robotic system layers
- **Quantitative Risk Scoring**: Generates numerical security scores for comparative analysis  
- **Visual Analytics**: Produces radar charts and heatmaps for intuitive risk visualization
- **Extensible Architecture**: Supports addition of new attack vectors and defense mechanisms
- **Empirical Validation**: Includes real-world case studies on commercial robotic platforms

### Key Features

- 🛡️ **Comprehensive Coverage**: 39 attack vectors and 35 defense measures across Physical, Sensor & Perception, Data Processing, Middleware, Decision-Making, Application, and Social Interface layers
- 📊 **Quantitative Metrics**: Weighted risk scores with statistical validation
- 🎯 **Practical Applicability**: Tested on 3 commercial robotic platforms
- 🔧 **Automated Assessment**: Scriptable evaluation pipeline for CI/CD integration

## 📁 Repository Structure
```
RISK_MAP/
├── data/ # Core datasets and matrices
│ ├── attacks_vs_defenses_normalised.csv # Main attack-defense matrix
│ ├── attack_weights.csv # Attack severity weights
│ ├── per_layer_scores.csv # Generated layer scores
│ └── *_implementation_status.csv # Robot-specific defense implementations
├── scripts/ # Analysis and visualization scripts
│ ├── score_RISK_MAP.py # Main scoring algorithm
│ ├── combined_radar.py # Multi-robot comparison charts
│ └── [additional analysis scripts]
├── figures/ # Generated visualizations
│ ├── [robot_name]/
│ │ ├── radar.pdf # Individual radar charts
│ │ └── heatmap_top10.png # Risk heatmaps
│ └── combined_radar.pdf # Comparative analysis
├── docs/ # Documentation
│ ├── methodology.md # Detailed methodology
│ ├── attack_taxonomy.md # Complete attack vector taxonomy
│ └── defense_catalog.md # Defense mechanism catalog
├── examples/ # Usage examples
│ ├── quick_start_example.py # Basic usage demonstration
│ └── custom_robot_assessment.py # Adding new robots
├── tests/ # Unit tests and validation
│ ├── test_scoring_algorithm.py # Algorithm validation
│ └── test_data_integrity.py # Data consistency checks
├── requirements.txt # Python dependencies
├── environment.yml # Conda environment specification
└── README.md # This file
```

## Installation

### Prerequisites
- Python **3.8+**
- Git
- JupyterLab or Jupyter Notebook

#### Option 1 — `venv` + pip (recommended)
```bash
git clone https://github.com/anonymous4sc1ence/submission.git
cd submission

python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows PowerShell:
# .\venv\Scripts\Activate.ps1

pip install -r requirements.txt


```
#### Option 2: Using conda
```
# Clone the repository
git clone https://github.com/anonymous4sc1ence/submission.git
cd submission

# Create conda environment
conda env create -f environment.yml
conda activate risk-map

```
#### Verify Installation
```
python scripts/score_RISK_MAP.py --help

```
### Quick Start
#### Assessment

Run RISK-MAP assessment on all robots in the dataset:
```
python scripts/score_RISK_MAP.py

```
#### This will:

  - Process all *_implementation_status.csv files in data/
  - Generate individual radar charts in figures/[robot_name]/
  - Create per-layer scores in data/per_layer_scores.csv
  - Display summary results in the terminal
  - Create combined visualization comparing multiple robots:
#### Expected Output
```
[✓]    Robot1: RISK_MAP  39.9%  → figures/Robot1
[✓]    Robot2: RISK_MAP  48.9%  → figures/Robot2  
[✓]    Robot3: RISK_MAP  79.5%  → figures/Robot3
[✓] Wrote per-layer scores → data/per_layer_scores.csv
[✓] Combined radar → figures/combined_radar.pdf

```

# Reproducing Paper Results
Main Experimental Results (Section 5)


> All reproduction steps below assume your shell is located at: `submission/1. RISK_MAP/`.



## What this artifact reproduces (from the paper)

- **Layer coverage radar plots** and **defense coverage heatmaps** for three humanoids (Digit, G1 EDU, Pepper), and  
- **Overall risk-weighted scores** with layer-wise diagnostics and **Top-10 residual risks for the 3 robots** .

These correspond to the quantitative RISK-MAP method and results reported in the paper (Sections 5–5.2; Figures 4a/4b; Table 3).



## Quick Start (Notebook-first, recommended)

1) **Create an environment** (pick one):
- **venv**
```
  ```bash
  python -m venv .venv
  # Windows:
  .venv\Scripts\activate
  # Linux/macOS:
  source .venv/bin/activate

```
  # Minimal packages commonly used in the artifact
```
  pip install jupyter pandas numpy matplotlib

```
Open the RISK-MAP notebook in this folder (the notebook provided in 1. RISK_MAP/) and Run All.
The notebook will:

 - Load the core matrices from data/attacks_vs_defenses_normalised.csv and data/attack_weights.csv.
 - Load robot implementation CSVs in data/ (e.g., *_implementation_status.csv).
 - Compute:
    -- Overall RISK-MAP score per robot
    -- 7-layer coverage (radar)
    -- Top-10 residual risks (heatmap)
    -- Write figures into figures/ (per-robot subfolders) and any combined plots (e.g., figures/RISK_MAP_combined_radar.pdf, figures/combined_heatmap_safe_r.png).

Note (Windows paths): If you run from OneDrive or paths containing spaces, the notebook/plots still work as long as you keep relative paths (as they are in the repo).
```
```
Expected runtime: ~5 minutes on standard desktop hardware.

📊 Dataset Description
Core Datasets
| File                                 | Description                         | Dimensions               | Purpose                |
| ------------------------------------ | ----------------------------------- | ------------------------ | ---------------------- |
| `attacks_vs_defenses_normalised.csv` | Attack-defense effectiveness matrix | 39 attacks × 35 defenses | Core assessment logic  |
| `attack_weights.csv`                 | Attack severity weights             | 39 attacks × 1 weight    | Risk prioritization    |
| `*_implementation_status.csv`        | Robot defense implementations       | 35 defenses × 1 status   | Individual assessments |

# Data Sources

Attack Vectors: Derived from systematic literature review of 89 robotics security papers (2015-2025)

Defense Mechanisms: Catalogued from commercial robotics frameworks and security standards

Effectiveness Ratings: Expert assessment validated through empirical testing

Severity Weights: Based on CVSS v3.1 and robotics-specific impact analysis

Data Quality Assurance

✅ Inter-rater reliability: κ = 0.82 (substantial agreement)

✅ Coverage validation: 100% mapping to robotics security taxonomy

✅ Consistency checks: Automated validation in tests/test_data_integrity.py

📈 Performance and Scalability

Assessment Speed: ~0.1 seconds per robot on standard hardware
Memory Usage: <100MB for datasets with 1000+ robots
Scalability: Linear complexity O(n) for n robots
Parallel Processing: Built-in support for batch assessments

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
```


