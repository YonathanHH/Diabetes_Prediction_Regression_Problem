# Diabetes_Prediction_Regression_Problem
This repository use artificial data of a diabetes predictor from kaggle to showcase end-to-end portolio project

# Online Deployable Streamlit App
[streamlit link](reservedlink)

# Project Overview

This project demonstrates an end-to-end machine learning workflow for healthcare analytics:

- **Problem Type:** Regression (predicting continuous diabetes probability 0-1)
- **Dataset Size:** 10,000 observations with 11 medical features
- **Best Model:** Tuned Gradient Boosting (selected during benchmarking)
- **Performance:** R² ≈ 0.99 on test set, MAPE: 0.00012
- **Deployment:** Streamlit web application

### Key Features
- Complete EDA with visualization  
- Advanced feature engineering with pipelines  
- 10 model benchmarking (including XGBoost, CatBoost, LightGBM)  
- Hyperparameter tuning with RandomizedSearchCV  
- Residual analysis & model validation  
- Feature importance analysis  
- Interactive Streamlit deployment  
- Production-ready code with documentation  

---

## Dataset Information

**Source:** [Kaggle - Comprehensive Dataset for Diabetes Prediction](https://www.kaggle.com/datasets/str1k3rfl0/comprehensive-dataset-for-diabetes-prediction)

### Features (11 columns)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `age` | Numerical | 18-100 | Patient's age in years |
| `gender` | Categorical | M/F | Patient's gender |
| `bmi` | Numerical | 10-60 | Body Mass Index |
| `blood_pressure_category` | Categorical | 4 classes | Normal, Hypertension stage I, Hypertension Stage II, Elevated |
| `cholesterol` | Numerical | 100-400 | Total cholesterol level (mg/dL) |
| `glucose` | Numerical | 70-200 | Blood glucose level (mg/dL) |
| **`diabetes_probability`** | **Numerical** | **0-1** | **Target variable** |

### Prerequisites

```bash
Python 3.8+
pip or conda
```

## Running Jupyter Notebook
It is advised to use VS Code to run the code.

## Model deployment testing - Option 1
Importing required library, dataset, and model
```
import pandas as pd
import pickle

df = pd.from_csv("hotel_dataset_cleaned.csv")
pipe = pickle.load(open("final_model.sav", "rb")) ### Openning the data
```
Testing example
```
print('predict class :',pipe.predict(df[51:55]))
print('predict proba :',pipe.predict_proba(df[51:55]))
```

## Model deployment using streamlit - Option 2 (Interactive)
- Step 1: Open python or conda terminal
- Step 2: Change into desired virtual environment
- Step 3: Change directory to desired ML folder
- Step 4: Run Streamlit as shown below
```
streamlit run app.py
```

## Required Libraries
```
- Matplotlib 3.0.2
- Numpy 2.3.5
- Pandas 2.3.3
- Sklearn (ScikitLearn) 1.7.2
- Shap 0.50.0
- Seaborn 0.13.2
- catboost 1.2.8
- lightgbm 4.6.0
- xgboost 3.1.2
- pickle (built-in)
- imbalanced-learn 0.14.0
```

## Key Insights & Recommendations

### Business Recommendations

1. **Deployment Strategy**
   - Integrate into healthcare provider EHR systems
   - Use as risk screening tool during consultations
   - Target high-risk patients for early intervention programs

2. **Maintenance Schedule**
   - Retrain model quarterly with new patient data
   - Monitor prediction accuracy vs actual outcomes
   - Update risk thresholds based on clinical feedback

3. **Usage Context**
   - Real-time risk assessment during patient visits
   - Population health screening
   - Preventive care resource allocation

### Model Recommendations

1. **Feature Engineering Enhancements**
   - Add temporal features (trend in blood glucose, BP)
   - Include lifestyle indicators (exercise, diet quality)
   - Incorporate family history and comorbidities

2. **Model Improvements**
   - Implement ensemble voting for robustness
   - Use stratified k-fold cross-validation
   - Experiment with feature interactions
   - Consider transfer learning from related datasets

3. **Interpretation**
   - SHAP force plots for individual predictions
   - Feature dependency analysis
   - Clinical validation with domain experts

### Data Recommendations

1. **Data Collection**
   - Include actual medical outcomes (diagnosed diabetes cases)
   - Temporal follow-up data
   - Lifestyle and behavioral features
   - Geographic and socioeconomic factors

2. **Data Quality**
   - Address class imbalance if target is binary
   - Validate medical ranges with health professionals
   - Regular data audits for outliers and errors
  
### Machine Learning Concepts Used

1. **Regression:** Predicting continuous values (probability 0-1)
2. **Feature Scaling:** StandardScaler for linear models
3. **Ensemble Methods:** Combining multiple models for better predictions
4. **Hyperparameter Tuning:** RandomizedSearchCV for optimization
5. **Cross-Validation:** k-fold for robust evaluation
6. **Pipeline:** Automating preprocessing and model training
