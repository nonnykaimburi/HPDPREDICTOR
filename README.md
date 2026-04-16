# HPDPREDICTOR

Hypertensive Disorder of Pregnancy Risk Prediction System - A clinical decision support system for predicting maternal hypertensive disorders using machine learning.

## Features

- Machine learning models for maternal risk prediction
- Firebase integration for data management
- Streamlit web interface for easy access
- SHAP and LIME explainability analysis
- Docker containerization support
- Cloud deployment ready (Streamlit Cloud, Render, Vercel)

## Quick Start

### Windows Users
Double-click `run.bat` to launch the app automatically.

### Mac/Linux Users
```bash
bash run.sh
```

### Manual Launch
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

This app is ready for deployment on:
- **Streamlit Cloud** - Recommended for easiest setup
- **Render** - See `render.yaml`
- **Vercel** - See `vercel.json` and `FIREBASE_VERCEL_SETUP.md`
- **Docker** - See `Dockerfile`

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Configure Firebase credentials (optional)
3. Run the application using one of the methods above

## License

MIT

---

## ✅ What Happens When You Run

1. ✅ The launcher automatically detects an available port (starting from 8501)
2. ✅ Initializes Streamlit with Firebase integration
3. ✅ Loads the trained Random Forest model
4. ✅ Displays access URL in the terminal

**Example output:**
```
============================================================================
             HDP PREDICTOR - Maternal Health Risk Prediction System
============================================================================

✅ Port 8503 is available
🚀 Starting HDP PREDICTOR on port 8503...
📱 Access the app at: http://localhost:8503

⏳ Initializing... (this may take 30-60 seconds)
```

Then open your browser to the displayed URL!

---

## 🔓 Login Credentials

- **Username:** `doctor`
- **Password:** `health123`

---

## 📊 Features

- **View Patients**: Browse the roster of 10 African patients with longitudinal hypertension data
- **Enroll New Patient**: Add new patients with initial visit vitals
- **Risk Prediction**: Get HDP risk probability and time-to-event estimates
- **System Architecture**: View model documentation, data sources, and clinical framework
- **Firebase Sync**: All patient data persists across sessions

---

## 🏥 Clinical Data Used

**Primary Sources:**
1. **UCI Maternal Health Risk Dataset** - M. Ahmed, 2020 (https://doi.org/10.24432/C5DP5D)
2. **CORIS Baseline Study** - Dr. Rossouw et al., South African Medical Journal Vol. 64, Issue 12

**Features:**
- Systolic/Diastolic Blood Pressure (mmHg)
- Heart Rate (bpm)
- Blood Glucose (mmol/L)
- BMI, Age, Prenatal Visits
- Comorbidities: Hypertension, Diabetes
- Pregnancy markers: Oliguria, Proteinuria

---

## 🤖 Model Performance

| Metric | Value |
|--------|-------|
| Classification Accuracy | 100.0% |
| F1 Score | 1.0000 |
| Time Prediction MSE | 0.2135 (±0.46 weeks) |

**Training:** 2,500 maternal health records (70/20/10 train/val/test split)

---

## 📁 Project Structure

```
HEALTH - Copy/
├── app.py                      # Main Streamlit app
├── train_model.py              # Model training script
├── launcher.py                 # Smart port finder & launcher
├── run.bat                     # Windows quick-start
├── run.sh                      # Mac/Linux quick-start
├── requirements.txt            # Dependencies
├── .streamlit/
│   └── config.toml             # Streamlit configuration
└── artifacts/
    ├── risk_classifier.pkl     # Trained classifier
    ├── time_regressor.pkl      # Trained regressor
    └── ...                     # Scaler data
```

---

## 🔧 Troubleshooting

**Q: Port errors even with launcher?**
A: The launcher automatically finds the next available port. If you still see errors, check if Streamlit/Python processes are stuck:
```powershell
taskkill /IM python.exe /F
```

**Q: FireBase not initialized?**
A: This is normal - the app works in offline mode. Patient data stores locally until Firebase credentials are configured.

**Q: For deployment to Vercel/Render?**
Set the `FIREBASE_SERVICE_ACCOUNT_BASE64` environment variable with your base64-encoded Firebase credentials.

---

## 📞 Support

For issues or questions, refer to the **System Architecture** page in the app for detailed model documentation and citations.

---

**Last Updated:** April 15, 2026  
**Status:** Production Ready ✅
