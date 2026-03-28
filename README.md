# DSAI 4103 Course Project - Loan Approval Prediction

**Finance Company Loan Eligibility Automation with Fairness Analysis**

[dashboard.pbix](https://github.com/rasha-2712/DSAI4103-Loan-Approval-Pipeline/blob/main/dashboard.pbix)
## 📌 Business Problem
A finance company wants to **automate loan approval** decisions to speed up processing, reduce risk, and ensure fairness across gender and property area groups.  
This project implements a complete end-to-end machine learning pipeline with strong focus on **model explainability (SHAP)** and **bias/fairness analysis (fairlearn)** — an angle not commonly explored in standard submissions.

## 🎯 Objectives
- Identify key factors affecting loan approval
- Build a robust predictive model using ensemble methods
- Provide SHAP-based explainability
- Conduct fairness analysis on protected attributes (Gender)
- Deliver an interactive Power BI dashboard for stakeholders

## 📊 Dataset
- **Source**: Analytics Vidhya – Loan Eligibility Prediction Competition
- **Files**: 3 CSV files (train, test, sample_submission)
- **Total columns across files**: 27+ (exceeds requirement)
- **Rows**: Train = 614, Test = 367
- **Key features**: ApplicantIncome, Credit_History, Property_Area, Gender (used for fairness)

## 🛠️ Methodology

### 1. Exploratory Data Analysis (EDA)
- Statistical summaries, missing value analysis, target distribution
- Visualizations saved for dashboard (approval rate, gender approval, income vs approval, correlation heatmap)

### 2. Data Preprocessing
- Missing value imputation (mode/median from train only)
- Label encoding for categorical features
- Standard scaling for numerical features
- Feature engineering: `Total_Income` and `LoanIncomeRatio`

### 3. Predictive Model
- **Model**: XGBoost Classifier (ensemble method)
- Validation results:
  - ROC-AUC: **0.7854**
  - Accuracy: **0.7724**

### 4. Explainability & Fairness
- **SHAP** (TreeExplainer): Summary plot + force plot
- **Fairlearn** bias analysis:
  - Demographic Parity Difference (Gender): **0.1441**
  - Equalized Odds Difference (Gender): **0.0976**

### 5. Deployment Preparation
- Model, scaler, and encoders saved using joblib
- Standalone `scoring_script.py` provided

### 6. Power BI Dashboard
- **3 interactive pages**: Overview, Insights, Predictions & Fairness
- KPIs, slicers, SHAP visualizations, fairness metrics, and detailed insights

## 📁 Repository Contents
- `Predicting loan eligibility.ipynb` → Full Jupyter notebook
- `scoring_script.py` → Ready-to-use prediction function
- `dashboard_data.csv` → Data exported for Power BI
- `loan_approval_model.pkl`, `scaler.pkl`, `label_encoders.pkl`
- All visualization PNGs (`shap_summary.png`, `approval_rate.png`, etc.)
- `Loan_Approval_Dashboard.pbix` → Final Power BI file

## 🚀 How to Run
1. Clone the repository
2. Open `Predicting loan eligibility.ipynb` in Colab or Jupyter
3. Run all cells (data is included)
4. Use `scoring_script.py` for new predictions:

```python
import pandas as pd
from scoring_script import predict_loan   # if you imported it

new_application = pd.DataFrame({
    'Gender': ['Female'],
    'Married': ['No'],
    'Dependents': ['0'],
    'Education': ['Graduate'],
    'Self_Employed': ['No'],
    'ApplicantIncome': [4500],
    'CoapplicantIncome': [0],
    'LoanAmount': [100],
    'Loan_Amount_Term': [360],
    'Credit_History': [1.0],
    'Property_Area': ['Urban']
})

prob = predict_loan(new_application)
print(f"Approval Probability: {prob[0]:.4f} ({prob[0]*100:.2f}%)")
