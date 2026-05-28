# Nano Particle Tracking Analysis (NTA) Web App Pro 🔬

Software for simulating the **Kalman Filter** algorithm to correct AI detection errors in microfluidic environments.

## 🚀 Architecture Workflow

Below is the architecture and processing flow of the application:

![NTA Workflow Architecture](images/architecture.png?v=2)

## 🛠 Installation & Usage Guide

1. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Access the app via browser, upload an NTA video, and configure parameters for analysis.

## ✨ Key Features

- Nano particle detection and tracking via OpenCV.
- Kalman Filter algorithm to predict and correct particle trajectories.
- Particle size calculation based on the Stokes-Einstein equation and Mean Squared Displacement (MSD).
- Real-time physics reporting charts.
