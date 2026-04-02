# House Price Predictor — India

Real-time ML-based house price prediction across **5 major Indian cities** and **25 micro-markets**, built with scikit-learn / XGBoost and a professional Streamlit UI.

huggingface deploy: https://huggingface.co/spaces/cherrisai/House_prediction

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2.Train model (saves model.pkl)
python train.py

# 3. Launch app
streamlit run app.py
```

> **Note:** Steps 2 & 3 are optional — `data.csv` and `model.pkl` are already included.

## Features

- **ML Model:** Random Forest (+ XGBoost if installed) — R² ≈ 0.95
- **Smart Pricing:** Synthetic data uses real avg ₹/sqft with area-level multipliers and BHK factors
- **Interactive Charts:** City comparison bars, area-level price variation, price distribution histogram
- **Dynamic UI:** Area dropdown auto-updates based on selected city

## Project Structure

```
├── app.py             # Streamlit UI
├── train.py           # Model training (RF + XGB)
├── generate_data.py   # Synthetic dataset generator
├── data.csv           # 3000-row training dataset
├── model.pkl          # Trained pipeline (preprocessor + model)
├── requirements.txt   # Dependencies
└── README.md
```

## Tech Stack

Python · Pandas · NumPy · scikit-learn · XGBoost · Plotly · Streamlit
