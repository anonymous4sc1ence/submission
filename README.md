# RISK-MAP: A Comprehensive Framework for Robotics Security Assessment

[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxxx-blue)](https://doi.org/10.xxxx/xxxxxx)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Artifact Submission for USENIX Security 2025**  
> Paper Title: [Your Paper Title]  
> Authors: [Anonymous for Review]

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

- 🛡️ **Comprehensive Coverage**: 50+ attack vectors across Physical, Sensor & Perception, Data Processing, Middleware, Decision-Making, Application, and Social Interface layers
- 📊 **Quantitative Metrics**: Weighted risk scores with statistical validation
- 🎯 **Practical Applicability**: Tested on 10+ commercial robotic platforms
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

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/anonymous4sc1ence/submission.git
cd submission

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
Option 2: Using conda
```
# Clone the repository
git clone https://github.com/anonymous4sc1ence/submission.git
cd submission

# Create conda environment
conda env create -f environment.yml
conda activate risk-map

```
Verify Installation
```
python scripts/score_RISK_MAP.py --help

```
⚡ Quick Start
Basic Assessment

Run RISK-MAP assessment on all robots in the dataset:
```
python scripts/score_RISK_MAP.py

```
This will:

Process all *_implementation_status.csv files in data/

Generate individual radar charts in figures/[robot_name]/

Create per-layer scores in data/per_layer_scores.csv

Display summary results in the terminal

Generate Comparative Analysis

Create combined visualization comparing multiple robots:
```
python scripts/combined_radar.py

```
Expected Output
```
[✓]    Robot1: RISK_MAP  39.9%  → figures/Robot1
[✓]    Robot2: RISK_MAP  48.9%  → figures/Robot2  
[✓]    Robot3: RISK_MAP  79.5%  → figures/Robot3
[✓] Wrote per-layer scores → data/per_layer_scores.csv
[✓] Combined radar → figures/combined_radar.pdf

```
📖 Detailed Usage
Assessing a New Robot

Create Implementation Status File:
```
