# 🕵️‍♀️ Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using **Machine Learning**.  
The main goal is to identify suspicious activity in financial transaction data and help minimize financial loss due to fraud.  
The dataset is highly imbalanced, making this a realistic and challenging problem in applied data science.

---

## 🚀 Project Overview

- **Objective:** Build a model to classify transactions as *fraudulent* or *legitimate*.
- **Dataset:** [Credit Card Fraud Dataset (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Problem Type:** Binary Classification (Supervised Learning)
- **Challenge:** Handling severe class imbalance where fraudulent cases are less than 0.2% of total transactions.

---

## 🧰 Tools & Technologies

- **Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost  
- **Techniques:** EDA, Feature Scaling, SMOTE, Cross Validation, ROC-AUC, Precision-Recall Curves

---

## 🔍 Workflow

1. **Data Exploration & Cleaning**
   - Identified missing values, outliers, and checked feature distributions.
   - Detected high imbalance between fraud and non-fraud classes.

2. **Feature Engineering**
   - Standardized numerical features using `StandardScaler`.
   - Applied dimensionality reduction (PCA) for better model performance.

3. **Imbalance Handling**
   - Used **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic fraud samples.
   - Compared results with and without SMOTE to assess its impact.

4. **Model Training**
   - Trained multiple models:
     - Logistic Regression  
     - Random Forest  
     - XGBoost
   - Evaluated each using AUC, Precision, Recall, and F1 Score.

5. **Model Evaluation**
   - Optimized for **Recall**, since missing a fraud (false negative) is more costly than false alarms.
   - Plotted **ROC-AUC** and **Confusion Matrix** for comparison.

---

## 📊 Model Performance

| Model | AUC Score | Key Notes |
|--------|------------|-----------|
| **Logistic Regression** | 🟢 **0.9719** | Best generalization; simple yet highly effective. |
| **XGBoost** | 🟢 **0.9648** | Performs nearly as well; handles complex non-linear patterns. |
| **Random Forest** | 🟡 **0.9310** | Strong performer but slightly less effective on imbalanced data. |

✅ **Best Model:** Logistic Regression (highest AUC with efficient computation).  
💡 **Insight:** Simpler linear models can outperform ensembles when preprocessing and feature scaling are well-tuned.

---

## 💡 Key Insights

- Handling **class imbalance** significantly boosts recall for fraud detection.
- Fraudulent transactions display distinct statistical patterns that ML models can capture effectively.
- Proper evaluation metrics (AUC, Recall) are crucial—accuracy alone is misleading for imbalanced data.

---

## 🧠 Future Improvements

- Experiment with **Autoencoders** or **Isolation Forest** for anomaly detection.
- Implement a **real-time prediction API** using `FastAPI` or `Flask`.
- Deploy model on **AWS / GCP / Streamlit Cloud** for live monitoring.
- Use **Explainable AI (SHAP)** for model interpretability.

---
