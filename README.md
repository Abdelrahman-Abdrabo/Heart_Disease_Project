# 🫀 Heart_Disease_Project

  
A machine learning pipeline for **heart disease detection** using the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).  

This project analyzes and predicts heart disease risks using machine learning. It covers the entire workflow from data preprocessing and dimensionality reduction (PCA) to training and evaluating various classification models (Logistic Regression, Decision Trees, Random Forest, SVM) and exploring unsupervised clustering methods.

  

---  

## 🌟 Overview

Cardiovascular disease is one of the leading causes of death globally.  

This project leverages **machine learning models** to predict the presence of heart disease based on patient health metrics such as age, cholesterol, blood pressure, chest pain type, and ECG results.

  

The pipeline:

- Automates **data preprocessing** and **feature engineering**.

- Explores **dimensionality reduction (PCA)** and **feature selection**.

- Compares multiple **classification models**.

- Optimizes performance with **hyperparameter tuning**.

- Deploys an interactive **Streamlit web application**.

  

---

  

## 📊 Dataset

- **Source**: [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)  

- **Samples**: 303 patients  

- **Features**: 13 raw attributes (numeric + categorical)  

- **Target**: Presence of heart disease (`0 = no disease`, `1 = disease`)

  

---

  

## 🔄 Project Workflow

  

### Phase 2.x Steps

1. **Data Preprocessing**
    - Handling missing values
    - Encoding categorical features (One-Hot Encoding)
    - Binarizing the target (`num > 0 → 1`)
    - Standardization (scaling)

  

2. **Dimensionality Reduction**
    - Principal Component Analysis (PCA)
    - Explained variance ratio & visualization

  

3. **Feature Selection**
    - Feature importance ranking
    - Final selected features (16 best predictors)


4. **Supervised Learning**
    - Logistic Regression, Decision Tree, Random Forest, SVM
    - Performance comparison using Accuracy, Precision, Recall, F1-score, AUC
  

5. **Unsupervised Learning**

    - KMeans clustering
    - Elbow method 

  

6. **Hyperparameter Tuning**
    - GridSearchCV & RandomizedSearchCV
    - Best model: **SVM (RBF kernel)** with optimized `C` and `gamma`

  

7. **Model Export**
    - End-to-end pipeline (preprocessing + feature selection + scaling + model)
    - Saved as `final_pipeline.pkl` with `joblib`

  

8. **Deployment (Streamlit UI)**
    - Simple form-based web interface
    - User inputs patient data
    - Model predicts `Disease` vs `No Disease` with probability

  

---

  

## 🧠 Model Development

- **Final selected features**:
```python
['exang_0.0','exang_1.0','slope_2.0','cp_4.0','cp_1.0', 'ca','slope_1.0','thal_3.0','oldpeak','age',   'sex_0.0','cp_3.0','thalach','trestbps','chol','thal_7.0']
```
- **Best model**: SVM with tuned hyperparameters (`C`, `gamma`)  
- **Pipeline**: `preprocessing → feature selection → scaling → SVM`

---

## 💻 Deployment (Streamlit UI)

### UI Features
- **Form-based input** for patient data (e.g., age, cholesterol, blood pressure).
- Clear categorical choices (*Male/Female*, *Chest pain type*, etc.).
- **Prediction result**:  
- `"No Disease"` ✅  
- `"Disease"` ❌  
- Probability score for confidence.

### Run the App
```bash
streamlit run app.py
