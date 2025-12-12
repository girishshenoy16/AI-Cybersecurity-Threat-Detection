# AI-Powered Cybersecurity Threat Detection

## Overview
Simple end-to-end project that:
1. Generates sample network-like data
2. Trains a RandomForest model to detect "Attack" vs "Benign"
3. Exposes a Flask API `/predict` that accepts JSON and returns threat detection
4. Includes instructions to run locally

This project was created based on user-provided project guide. See project document reference. fileciteturn0file0

## Requirements
- Python 3.8+
- pip install -r requirements.txt

## Files
- generate_sample_data.py : creates synthetic dataset `sample_cyber_data.csv`
- train_model.py : preprocesses, trains, saves `cybersecurity_model.pkl`
- app.py : Flask API for prediction
- utils.py : helper functions for preprocessing & feature vector creation
- requirements.txt : Python dependencies
- submission_report.md : short project report for internship submission

## Quick start (local)
1. Create virtualenv: `python -m venv venv && source venv/bin/activate` (Windows: `venv\Scripts\activate`)
2. Install: `pip install -r requirements.txt`
3. Generate sample data: `python generate_sample_data.py --out sample_cyber_data.csv --n 5000`
4. Train model: `python train_model.py --data sample_cyber_data.csv --out cybersecurity_model.pkl`
5. Run API: `python app.py`
6. Test with curl or Postman to `http://127.0.0.1:5000/predict` with JSON like:

{
  "packet_size": 1500,
  "failed_logins": 5,
  "request_frequency": 250,
  "src_bytes": 12345,
  "dst_bytes": 54321,
  "duration": 0.5
}