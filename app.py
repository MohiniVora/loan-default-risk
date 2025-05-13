from flask import Flask, render_template, request, redirect, url_for, session 
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Load model and preprocessor
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Home route - form
@app.route('/')
def home():
    prediction = session.pop('prediction', None)  # Get prediction from session if available
    tips = session.pop('tips', None)
    proba = session.pop('proba', None)
    return render_template('form.html', prediction=prediction, tips=tips, proba=proba)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect form data
        features = [
            float(request.form['InterestRate']),
            float(request.form['LoanTerm']),
            float(request.form['DTIRatio']),
            int(request.form['NumCreditLines']),
            request.form['Education'],
            request.form['EmploymentType'],
            request.form['MaritalStatus'],
            request.form['HasMortgage'],
            request.form['HasDependents'],
            request.form['LoanPurpose'],
            request.form['HasCoSigner'],
        ]
        
        # Split into numerical and categorical manually
        num_features = np.array(features[:4]).reshape(1, -1)
        cat_features = np.array(features[4:]).reshape(1, -1)

        # Combine
        
        input_df = pd.DataFrame(np.hstack((num_features, cat_features)),
                                columns=['InterestRate', 'LoanTerm', 'DTIRatio', 'NumCreditLines', 
                                         'Education', 'EmploymentType', 'MaritalStatus', 
                                         'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'])
        
        # Preprocess input
        input_processed = preprocessor.transform(input_df)
        
        # Predict
        # Get probability of default
        proba = model.predict_proba(input_processed)[0][1]  # Class 1: Default
        risk_percent = round(proba * 100, 2)

        # Determine prediction message
        if proba >= 0.5:
            result = f"⚠️ High Risk: Likely to Default ({risk_percent}% risk)"
        else:
            result = f"✅ Low Risk: Likely to Repay ({100 - risk_percent}% confidence)"

        # Personalized tips
        tips = []
        if float(request.form['DTIRatio']) > 0.4:
            tips.append("💡 Try lowering your DTI Ratio below 0.4 to reduce risk.")
        if float(request.form['InterestRate']) > 15:
            tips.append("💡 A lower interest rate can improve approval chances.")
        if request.form['EmploymentType'] == "Unemployed":
            tips.append("💡 Stable income can increase approval probability.")
        if int(request.form['NumCreditLines']) <= 2:
            tips.append("💡 Building a longer credit history may help.")


        session['prediction'] = result  # Store prediction in session
        session['tips'] = tips
        session['proba'] = proba

        # Combine
        return render_template("form.html", prediction=result, tips=tips, proba=proba)

        
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
