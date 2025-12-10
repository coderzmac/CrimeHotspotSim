# CrimeHotspotSim 🔍🗺️
A spatio-temporal hotspot forecasting and simulation tool that predicts weekly crime risk in Cleveland, Ohio, and models how interventions—such as increased police patrols or pedestrian activity—may reshape future hotspot patterns.

## Overview
CrimeHotspotSim is a research-driven project that leverages Cleveland’s open crime API (2016–present), along with Census demographics, environmental data (temperature), and satellite nightlight imagery, to forecast crime hotspots at a 0.5 km × 0.5 km grid resolution.
The project’s main contribution is the ability to **simulate interventions** and generate **counterfactual hotspot maps** that estimate how predicted crime risk shifts under different scenarios. 

Examples include:
- Increasing pedestrian activity by 30%
- Increasing police patrol presence in hotspot areas by 40%
- Testing combined interventions to visualize shifts in spatial crime risk

This approach provides **policymakers**, **law enforcement**, and **researchers** with a way to explore strategies virtually before committing real-world resources.

---

## Features
- **Full preprocessing pipeline** for Cleveland crime data via ArcGIS FeatureServer
(timestamp cleaning, invalid coordinate removal, district filtering, 2016+ records).
- **Spatial gridding** into 0.5 km cells and **weekly temporal aggregation**.
- **Spatio-temporal feature engineering**, including:
  - Lag features (lag1, lag2, lag3)
  - Spatial neighbor features (1-week, 4-week, 8-week averages)
  - ACS Census demographics (income, poverty, unemployment)
  - Environmental features (temperature, nightlight)
- **Machine learning forecasting models**:
  - GradientBoostingClassifier 
  - XGBoostClassifier 
- **Counterfactual intervention simulator**, supporting:
  - Pedestrian activity increases/decreases 
  - Police patrol adjustments for hotspot vs. non-hotspot grids 
- **Exportable predictions**, including:
  - Baseline hotspot probabilities 
  - Post-intervention adjusted risk maps 
- Processed datasets for downstream analysis, modeling, or visualization (e.g., Kepler.gl, Folium, GIS workflows).

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

---

## Usage

### 1. Prepare the data directory
Place the required raw data files into the project structure:

```
data/ 
 ├── raw/
 │    ├── temperature.csv          # Daily temperature data (NOAA or equivalent)
 │    └── (automatically pulls Cleveland crime data via API)
 ├── external/
 │    └── night_light.tif          # (Optional) VIIRS nightlight raster
 └── processed/
      (will be created automatically)
```

---

### 2. Run the preprocessing pipeline
This fetches Cleveland crime data from the ArcGIS API, builds the spatial grid,
aggregates by week, creates lag + neighbor features, joins Census demographics,
and outputs the core modeling datasets.

```bash
python src/preprocessing.py --outdir data/processed
```

---

### 3. Train a baseline Gradient Boosting model

This trains a hotspot classifier, prints validation metrics + feature importances,
and generates before/after intervention predictions.

```bash
python models/modeling.py --frames data/processed/frames.csv --models_dir models --pred_dir predictions
```

---

### 4. Train the XGBoost model (optional, more performant)

This creates the XGBoost baseline model and runs the same intervention simulation.

```bash
python models/xgboost_modeling.py --frames data/processed/frames.csv --models_dir models --pred_dir predictions
```

---

### 5. Visualize results (Kepler.gl, Folium, GIS tools)

You can load:
- latest_scores_before.csv
- latest_scores_after.csv
- xgb_latest_scores_after.csv 

into Kepler.gl, QGIS, or any GIS/mapping tool to visualize:
- Predicted hotspot probabilities 
- How risk shifts after interventions 
- Spatial cluster patterns and temporal changes