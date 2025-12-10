# CrimeHotspotSim 🔍🗺️
A spatio-temporal hotspot mapping tool that forecasts where crime is most likely to occur in Los Angeles and simulates the impact of preventative interventions.

---

## Overview
CrimeHotspotSim is a research-driven project that leverages crime data (2020–present) to predict high-risk areas in Los Angeles.
The project’s main contribution is the ability to **simulate interventions** (for example, adding 10 patrol cars to a grid, or increasing pedestrian activity by 5%) and generate **counterfactual hotspot maps** that estimate how predicted crime risk shifts under different scenarios.

This approach provides policymakers, law enforcement, and researchers with a way to explore strategies virtually before implementing them in real life.

---

## Features
- Preprocess large-scale crime datasets for analysis and modeling.
- Categorize raw crime descriptions into standardized groups (e.g., THEFT, ASSAULT, FRAUD).
- Engineer **spatio-temporal features** (Year, Month, Hour, DayOfWeek).
- Visualize hotspots on an **interactive Los Angeles map** using Folium.
- Simulate preventative interventions (patrol allocation, pedestrian activity).
- Export cleaned datasets for further modeling or research.

---

## Project Structure

```
CrimeHotspotSim/
│
├── data/ # Raw & processed datasets (ignored in Git)
│ ├── external/ # Place light pollution file here (e.g., night_light.tif)
│ ├── raw/ # Place original CSVs here (e.g., temperature.csv)
│ └── processed/ # Outputs after preprocessing (e.g., frames.csv, inference_frame.csv)
│
├── src/ # Python source code
│ └── preprocessing.py # Cleaning & feature engineering
│
├── models/ # Machine learning training and trained files
│ ├── modeling.py # Training & evaluation using GradientBoostingClassifier
│ ├── xgboost_modeling.py # Training & evaluation using XGBoostingClassifier
│ ├── baseline_ml.pkl # Trained GradientBoostingClassifier model
│ └── xgboost_modeling.py # Trained XGBoostingClassifier model
│
├── predictions/ # Csv files from ml models before and after risk scores are altered e
│ ├── latest_scores_after.csv # Provides features and risk score before model alterations
│ ├── latest_scores_before.csv # Altered features and risk score after GradientBoostingClassifier
│ └── xgboost_latest_scores_after.csv # Altered features and risk score after XGBoostingClassifier
│
├── .gitignore # Ensures raw data is not uploaded to GitHub
├── requirements.txt # Python dependencies
└── README.md # Project documentation

```

---

## Installation
Clone the repository and install dependencies:

```bash
git clone git@github.com:cOderZmAk/CrimeHotspotSim.git
cd CrimeHotspotSim

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # On Mac/Linux
.venv\Scripts\activate         # On Windows

# Install dependencies
pip install -r requirements.txt
```
## Usage
1. Place the raw dataset (`crime.csv`) into `data/raw/`.
2. Open JupyterLab:
   ```bash
   jupyter lab

Run the notebooks in order:
preprocessing.ipynb → cleans data and creates processed parquet file.
hotspot_mapping.ipynb → generates LA crime hotspot visualizations.
simulation.ipynb → runs counterfactual experiments (e.g., add patrols).
Check visualizations/ for exported maps (.html) you can open in a browser.


---
