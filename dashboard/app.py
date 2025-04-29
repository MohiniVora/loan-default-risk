from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Home route - form
@app.route('/')
def home():
    return render_template('form.html')

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
        import pandas as pd
        input_df = pd.DataFrame(np.hstack((num_features, cat_features)),
                                columns=['InterestRate', 'LoanTerm', 'DTIRatio', 'NumCreditLines', 
                                         'Education', 'EmploymentType', 'MaritalStatus', 
                                         'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'])
        
        # Preprocess input
        input_processed = preprocessor.transform(input_df)
        
        # Predict
        prediction = model.predict(input_processed)[0]
        
        # Map prediction to text
        if prediction == 1:
            result = "⚠️ High Risk: Likely to Default"
        else:
            result = "✅ Low Risk: Likely to Repay"

        return render_template('form.html', prediction=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
