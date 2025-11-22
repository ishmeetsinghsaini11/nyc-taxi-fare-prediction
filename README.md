
# 🗽 New York City Taxi Fare Prediction

This repository focuses on predicting **New York City taxi fares** using machine learning regression models.  
It includes complete EDA, feature engineering, model training, and evaluation workflows.

---

## 📊 Model Results (Kaggle Public Scores)

| Model | Description | RMSE (Public Score) |
|--------|--------------|--------------------|
| Linear Regression | Baseline model (no feature engineering) | 9.40717 |
| Ridge Regression | No hyperparameter tuning, some feature engineering | 5.50722 |
| Random Forest | max_depth=10, tuned features | **3.88037** |
| XGBoost | No hyperparameter tuning | 4.07514 |
| KNN | KNN model | 4.44475 |
| Decision Tree | Decision tree regressor | 6.03126 |
| SVM (SVR) | Linear SVR | 5.65757 |

✅ **Best Model:** Random Forest Regressor (RMSE: 3.88)

---

## 📘 Project Overview

This project explores and models the **New York City Taxi Fare Prediction** dataset from Kaggle.  
The pipeline is organized into two key notebooks:

- **(EDA)New_York_City_Taxi_Fare_Prediction.ipynb** → Data cleaning, feature extraction, and visualization.  
- **(model_train_and_evaluation)New_York_City_Taxi_Fare_Prediction.ipynb** → Model training, hyperparameter testing, and comparison.

---

## 🧩 Dataset

Dataset link: [Kaggle – New York City Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)

**Main columns:**
- `pickup_datetime`
- `pickup_longitude`, `pickup_latitude`
- `dropoff_longitude`, `dropoff_latitude`
- `passenger_count`
- `fare_amount` (target)

---

## ⚙️ Setup Instructions

```bash
git clone <git@github.com:ishmeetsinghsaini11/nyc-taxi-fare-prediction.git>
cd New_York_City_Taxi_Fare_Prediction

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

If `requirements.txt` is missing, install key dependencies manually:

```
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
jupyter
joblib
```

---

## 🔍 Workflow Summary

1. **EDA & Preprocessing:**
   - Removed outliers and invalid coordinates.
   - Computed distance using the Haversine formula.
   - Extracted datetime features (hour, day, month).
   - Visualized fare distributions and correlations.

2. **Feature Engineering:**
   - Added trip distance, direction, passenger buckets.
   - Normalized continuous variables.

3. **Model Training & Evaluation:**
   - Baseline Linear Regression → Ridge Regression → Tree-based models.
   - Evaluated using RMSE and MAE.

---

## 📈 Results Interpretation

| Model | RMSE | Key Insight |
|--------|------|-------------|
| Linear Regression | 9.40 | Simple baseline, underfits data |
| Ridge Regression | 5.50 | Regularization improves baseline |
| Random Forest | **3.88** | Best performer, captures non-linearities |
| XGBoost | 4.07 | Strong, needs tuning |
| KNN | 4.44 | Works decently with feature scaling |
| Decision Tree | 6.03 | High variance, overfitting risk |
| SVM (SVR) | 5.65 | Expensive, moderate results |

---

## 🧠 Key Learnings

- Feature engineering significantly improves model accuracy.
- Ensemble models (Random Forest, XGBoost) outperform linear baselines.
- RMSE reduced from **9.40 → 3.88**, showing strong iterative progress.

---


## 🚀 Next Steps

- Fine-tune XGBoost and LightGBM models.
- Add cross-validation and feature importance plots.
- Save serialized models for deployment (FastAPI/Flask).
- Explore deep learning models (DNN Regressor).

---

**Author:** Ishmeet Singh Saini  
**Project Type:** Machine Learning Regression (Kaggle)  
**Best Model:** Random Forest (RMSE: 3.88037)
