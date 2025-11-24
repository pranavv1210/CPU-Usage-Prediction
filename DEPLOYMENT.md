# Deployment Guide

This guide explains how to run the CPU Usage Prediction application locally.

## Prerequisites

- Python 3.8+
- pip

## Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Data & Train Model:**
    (Only needed if `model.joblib` is missing or you want to retrain)
    ```bash
    python generate_dummy_data.py
    python train_model.py
    ```
    *Note: We use `generate_dummy_data.py` because the original dataset is not accessible without DVC remote credentials.*

## Running the Application

1.  **Start the Flask Server:**
    ```bash
    python app.py
    ```

2.  **Access the App:**
    Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Application Structure

- `app.py`: The Flask application serving the model.
- `templates/index.html`: The modern, responsive frontend.
- `model.joblib`: The trained machine learning model.
- `train_model.py`: Script to train the model.
- `generate_dummy_data.py`: Script to generate sample data for demonstration.
