# 🎵 Spotify Stream Prediction

A Machine Learning web application that predicts the **estimated Spotify stream count** of a song using its audio features, popularity, release information, artist details, and other track characteristics.

The project uses an **XGBoost Regression Model** and **Streamlit** for the web interface.

---

## 📁 Project Structure

```text
spotify-stream-prediction/
│
├── app.py
├── spotify_stream_prediction_model.pkl
├── spotify_preprocessor.pkl
└── README.md
```

### Files

* `app.py` → Main Streamlit application
* `spotify_stream_prediction_model.pkl` → Trained XGBoost model
* `spotify_preprocessor.pkl` → Saved preprocessing pipeline
* `README.md` → Project documentation

---

## ⚙️ Setup & Installation

### 1. Open the project folder

Open the project folder in **VS Code**.

Open:

```text
Terminal → New Terminal
```

---

### 2. Check Python

```bash
python --version
```

Make sure Python is installed.

---

### 3. Install required libraries

```bash
pip install streamlit pandas numpy joblib xgboost scikit-learn==1.6.1
```

> **Important:** `scikit-learn==1.6.1` is required for compatibility with the saved model files.

---

### 4. Run the application

Make sure the terminal is inside the project folder.

Run:

```bash
streamlit run app.py
```

---

### 5. Open the application

After Streamlit starts, open:

```text
http://localhost:8501
```

The Spotify Stream Predictor will open in your browser.

---

## 🔮 How to Use

1. Enter the song information.
2. Enter the audio features.
3. Select the release date.
4. Enter artist and track details.
5. Click **Predict Spotify Streams**.
6. The application will display the estimated stream count.

---

## 🤖 Model Overview

**Algorithm:** XGBoost Regressor
**Training Data:** 2020–2024
**Evaluation Data:** 2025
**Target:** `log_stream_count`
**R² Score:** `0.5350`

The model predicts the log-transformed stream count and then converts it back to the estimated number of streams.

---

## 🛑 Stop the Application

To stop the Streamlit server:

```text
Ctrl + C
```

---

## 🚀 Quick Run

If everything is already installed:

```bash
cd path\to\spotify-stream-prediction
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

**That's it! 🎵**
