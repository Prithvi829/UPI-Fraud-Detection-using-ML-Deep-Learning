Smart UPI Fraud Detection System

A real-time UPI fraud detection system using Machine Learning and Deep Learning techniques such as Isolation Forest and Autoencoder.

Features

Detects anomalous UPI transactions
Real-time scoring API
Dashboard visualization for monitoring
Hybrid model approach (Isolation Forest + Autoencoder)

Tech Stack

Python
Flask
Scikit-learn
PyTorch

How it Works

Transaction data is processed through a feature pipeline
Isolation Forest and Autoencoder analyze transaction patterns
An anomaly score is generated
Results are served via API and displayed on the dashboard

How to Run

python run.py

Then open in browser:
http://127.0.0.1:5000

Project Structure

api/ backend routes
pipeline/ data processing and training
models/ machine learning models
dashboard/ frontend UI
run.py main entry point

Use Case

This system helps detect suspicious or fraudulent UPI transactions in real time, useful for fintech platforms and payment monitoring systems.
