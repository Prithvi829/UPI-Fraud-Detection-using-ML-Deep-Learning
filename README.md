UPI Payment Operations — Anomaly Detection System

Real-time UPI transaction anomaly detection system designed for payment operations, combining machine learning and deep learning models with a live monitoring dashboard and self-healing pipeline.

Overview

This system detects suspicious UPI transactions using a hybrid anomaly detection approach. It processes transaction data, applies feature engineering, and evaluates each transaction using two models:

Isolation Forest (tree-based anomaly detection)
Autoencoder (neural network reconstruction error)

The system exposes a REST API for real-time scoring and provides a dashboard for monitoring anomalies, risk levels, and system performance.

Core Capabilities

Real-time transaction scoring via API
Dual-model anomaly detection (Isolation Forest + Autoencoder)
Risk-based routing (LOW, MEDIUM, HIGH, CRITICAL)
Live dashboard with transaction feed and analytics
Drift detection with automatic model retraining
Thread-safe model registry with hot-swapping

Architecture

Transaction → Feature Engineering → Model Scoring → Risk Classification → API → Dashboard

Models operate in parallel:

Isolation Forest produces anomaly score
Autoencoder produces reconstruction error
Combined logic determines final risk level

How to Run

Install dependencies:

pip install -r requirements.txt

Run the system:

python run.py

Open in browser:

http://localhost:5001

On first run, the system:

processes data
trains models
starts API and dashboard

Subsequent runs skip training and start immediately.

API Endpoints

POST /api/v1/score
Scores a transaction and returns anomaly scores and risk level

GET /api/v1/transactions
Returns recent transaction log

GET /api/v1/analytics/summary
Returns aggregated statistics

GET /api/v1/analytics/timeseries
Returns anomaly trends over time

GET /api/v1/health
Returns system health and drift status

Example Request

POST /api/v1/score

{
"amount": 15000,
"hour_of_day": 2,
"day_of_week": 6,
"merchant_category": "P2P",
"txn_velocity_1h": 12,
"upi_id": "user@okicici",
"v_features": [0.0]
}

Tech Stack

Python
Flask
Scikit-learn
PyTorch
Plotly (dashboard)

Dependencies listed in

Configuration

All system parameters and paths are centralized in

Includes:

model paths
feature definitions
anomaly thresholds
drift detection parameters
Flask configuration

Project Structure

api/ REST API and routing
pipeline/ data processing, training, drift detection
models/ trained models and scalers
dashboard/ frontend UI
config.py configuration and hyperparameters
run.py system entry point

Dashboard Preview

8

The dashboard displays:

live transaction feed
anomaly rate over time
risk distribution
model performance metrics

Key Design Decisions

Hybrid model approach improves detection robustness
Consensus-based risk reduces false positives
Drift watchdog maintains model reliability over time
Centralized configuration avoids hardcoded values
Modular pipeline allows independent retraining

Use Case

Designed for fintech systems to:

detect fraudulent or suspicious transactions
assist payment operations teams
reduce manual review effort
monitor transaction risk in real time

Resume Summary

Real-time UPI fraud detection system using Isolation Forest and Autoencoder with a self-healing ML pipeline, REST API, and live monitoring dashboard.
