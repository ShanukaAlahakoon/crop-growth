# 🌱 Crop Growth Status Prediction Model

This repository contains the Machine Learning project for predicting the Growth Status (Good or Poor) of various crops based on environmental conditions, soil chemistry, and crop type.

The project follows a standard ML pipeline, culminating in the selection of a highly accurate XGBoost Classifier model for deployment.

---

## 🎯 Project Goal

The primary objective is to develop a robust binary classification model that accurately predicts whether a crop will achieve "Good Growth" (1) or "Poor Growth" (0) given a set of input parameters. This is intended to provide actionable insights for optimizing agricultural conditions.

---

## ✨ Features

### Machine Learning Model

- Uses **XGBoost** for predicting crop growth status.
- Model trained on features like **PH Value**, **Potassium (ppm)**, **Phosphorus (ppm)**, **Soil Type**, and environmental conditions like **Temperature**, **Humidity**, and **Sunlight hours**.

### Flask API

- The backend is built using **Flask**, which exposes an API endpoint `/predict` for making predictions based on user input.

### Front-End

- A modern, user-friendly **form-based interface** built using **Tailwind CSS**.
- **Dropdowns** for categorical inputs like Crop Type and Soil Type.
- **Number input fields** for continuous values like PH Value, Potassium, Phosphorus, Sunlight Hours, Temperature, and Humidity.

---

## 📊 Dataset and Features

Source: `component/dataset/crop_growth.csv` (included in this repo).

Key features used by the model:

- PH Value
- Potassium (ppm)
- Phosphorus (ppm)
- Sunlight Hours
- Temperature (°C)
- Humidity (%)
- Crop type (categorical)
- Soil Type (categorical)

Target: `Growth Status (Output)` (binary label used for supervised learning).

---

## ⚙️ Methodology (ML Pipeline)

### 1. Data Pre-processing & Feature Engineering

- Dropped irrelevant `ID` column.
- Categorical columns (`Crop type`, `Soil Type`) handled with `OneHotEncoder`.
- Numerical columns standardized with `StandardScaler`.
- Used a `ColumnTransformer` (preprocessor) to combine numeric scaling and categorical encoding.
- Train/test split performed (typical split: 80/20). Use stratified split if class imbalance is present.

### 2. Model Training & Evaluation

Models trained and evaluated in the notebook:

- XGBoost (`XGBClassifier`)
- Random Forest (`RandomForestClassifier`)
- Logistic Regression (`LogisticRegression`)
- Decision Tree (`DecisionTreeClassifier`)

Evaluation metrics used: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC / AUC, and 5-fold cross-validation scores.

### 3. Model Performance & Selection (Best Model: XGBoost)

The notebook trains and compares the models; XGBoost was selected as the final model due to superior AUC and F1-Score on the validation/test set (see `component/notebook/crop_growth.ipynb` for exact numbers and plots).

| Model               | Accuracy |  AUC | F1-Score (Class 1 - Good Growth) |
| ------------------- | -------: | ---: | -------------------------------: |
| XGBoost             |   85.00% | 0.88 |                             0.76 |
| Random Forest       |   81.67% | 0.88 |                             0.69 |
| Logistic Regression |   73.33% | 0.80 |                             0.50 |
| Decision Tree       |   70.00% | 0.72 |                             0.61 |

_Note_: Exact metric values and plots are in the notebook; open `component/notebook/crop_growth.ipynb` to inspect training logs and evaluation charts.

---

## 🔑 Key Feature Insights

Based on the trained XGBoost model's feature importance analysis, the top 5 features that most influence growth prediction are:

1.  PH Value
2.  Temperature (°C)
3.  Potassium (ppm)
4.  Humidity (%)
5.  Sunlight Hours

These rankings are indicative; re-training with more data or using SHAP explanations will provide deeper insight.

---

## 📦 Installation and Setup

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/ShanukaAlahakoon/crop-growth.git
cd crop-growth/backend
```

### 2. Set up Virtual Environment

It's recommended to use a virtual environment to keep your dependencies isolated:

**On Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On Mac/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

- **Flask** — Backend API framework
- **Joblib** — For loading and saving machine learning models
- **XGBoost** — The ML model used for predictions
- **Pandas** — For data handling
- **Scikit-learn** — For preprocessing and model utilities

### 4. Run the Flask App

Run the application locally:

```bash
python app.py
```

The Flask development server will start at `http://127.0.0.1:5000/`.

Open your browser and navigate to `http://127.0.0.1:5000/` to access the web interface.

---

## 🚀 Deployment and Usage

Saved deployment artifacts (as produced by the notebook and used by the Flask app):

- `best_growth_model.joblib` — trained XGBoost model (artifact in `component/models/`)
- `preprocessor.joblib` — `ColumnTransformer` that performs scaling + encoding
- `feature_names.joblib` — list of feature names (used to build the input DataFrame in correct order)

If you prefer alternative names, the commonly used names in other contexts are: `best_crop_growth_model.joblib` (model), `scaler.joblib` (numeric scaler), and `feature_names.joblib` (feature list). The repository currently uses the filenames listed above — adjust the names if you rename files.

---
