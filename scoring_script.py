
import pandas as pd
import joblib

# Load the saved artifacts
model = joblib.load('loan_approval_model.pkl')
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('label_encoders.pkl')

def predict_loan(new_data: pd.DataFrame):
    """
    Predict loan approval probability for new application(s).
    new_data must contain the same columns as the training data.
    """
    # Encode categorical columns
    for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
        new_data[col] = le_dict[col].transform(new_data[col].astype(str))
    
    # Scale numeric columns
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    new_data[num_cols] = scaler.transform(new_data[num_cols])
    
    # Return probability of approval (class 1)
    prob = model.predict_proba(new_data)[:, 1]
    return prob
