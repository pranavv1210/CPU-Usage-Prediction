# Cloud Deployment Guide (Render)

This guide will help you deploy your CPU Usage Prediction app to **Render** (a free cloud hosting platform).

## 1. Prepare Your Repository

We have already:
- Added `gunicorn` to `requirements.txt` (required for production).
- Created a `Procfile` (tells Render how to run the app).
- Un-ignored `model.joblib` so it can be uploaded to GitHub.

## 2. Push to GitHub

You need to push these changes to your GitHub repository. Run these commands in your terminal:

```bash
# Stop the running server first (Ctrl+C) if needed

# Add all new files
git add .

# Commit changes
git commit -m "Prepare for cloud deployment"

# Push to GitHub
git push origin main
```

## 3. Deploy on Render

1.  Go to [https://render.com/](https://render.com/) and sign up/login.
2.  Click **"New +"** and select **"Web Service"**.
3.  Connect your GitHub account and select your `CPU-Usage-Prediction` repository.
4.  Configure the service:
    - **Name:** `cpu-usage-predictor` (or any name you like)
    - **Region:** Choose the one closest to you.
    - **Branch:** `main`
    - **Root Directory:** (Leave blank)
    - **Runtime:** `Python 3`
    - **Build Command:** `pip install -r requirements.txt`
    - **Start Command:** `gunicorn app:app`
5.  Select the **"Free"** plan.
6.  Click **"Create Web Service"**.

Render will now build your app. It might take a few minutes. Once done, it will give you a URL (e.g., `https://cpu-usage-predictor.onrender.com`) where your app is live!
