from flask import Flask, render_template, request, Markup, make_response
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from xhtml2pdf import pisa
import io

app = Flask(__name__)

model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

def render_pdf(template_name, context):
    html = render_template(template_name, **context)
    result = io.BytesIO()
    pdf = pisa.pisaDocument(io.BytesIO(html.encode("UTF-8")), result)
    if not pdf.err:
        return result.getvalue()
    return None

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
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

        num_features = np.array(features[:4]).reshape(1, -1)
        cat_features = np.array(features[4:]).reshape(1, -1)

        input_df = pd.DataFrame(np.hstack((num_features, cat_features)),
                                columns=['InterestRate', 'LoanTerm', 'DTIRatio', 'NumCreditLines', 
                                         'Education', 'EmploymentType', 'MaritalStatus', 
                                         'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'])

        input_processed = preprocessor.transform(input_df)

        proba = model.predict_proba(input_processed)[0][1]
        risk_percent = round(proba * 100, 2)

        if proba >= 0.5:
            result = f"⚠️ High Risk: Likely to Default ({risk_percent}% risk)"
        else:
            result = f"✅ Low Risk: Likely to Repay ({100 - risk_percent}% confidence)"

        tips = []
        if float(request.form['DTIRatio']) > 0.4:
            tips.append("💡 Try lowering your DTI Ratio below 0.4 to reduce risk.")
        if float(request.form['InterestRate']) > 15:
            tips.append("💡 A lower interest rate can improve approval chances.")
        if request.form['EmploymentType'] == "Unemployed":
            tips.append("💡 Stable income can increase approval probability.")
        if int(request.form['NumCreditLines']) <= 2:
            tips.append("💡 Building a longer credit history may help.")

        explainer = shap.Explainer(model)
        shap_values = explainer(input_processed)

        shap_df = pd.DataFrame({
            "feature": input_df.columns,
            "shap_value": shap_values.values[0]
        }).sort_values(by="shap_value", key=abs, ascending=False).head(3)

        explanation = [
            f"{row['feature']}: {'⬆️' if row['shap_value'] > 0 else '⬇️'} Impact"
            for _, row in shap_df.iterrows()
        ]

        fig, ax = plt.subplots()
        shap_df.set_index("feature")["shap_value"].plot(kind='barh', ax=ax, color='skyblue')
        ax.set_title("Top SHAP Feature Impacts")
        fig.tight_layout()

        plot_path = os.path.join("static", "shap_plot.png")
        plt.savefig(plot_path)
        plt.close()

        os.makedirs("data", exist_ok=True)
        log_data = input_df.copy()
        log_data["Prediction"] = "High Risk" if proba >= 0.5 else "Low Risk"
        log_data["RiskScore"] = risk_percent
        log_data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path = "data/predictions_log.csv"
        if not os.path.exists(log_path):
            log_data.to_csv(log_path, index=False)
        else:
            log_data.to_csv(log_path, mode='a', header=False, index=False)

        return render_template("form.html", prediction=result, tips=tips, explanation=explanation, shap_plot=plot_path, form_data=input_df.iloc[0].to_dict())

@app.route('/dashboard')
def dashboard():
    log_path = "data/predictions_log.csv"
    if not os.path.exists(log_path):
        return "No data available yet."

    df = pd.read_csv(log_path)
    total = len(df)
    high_risk = len(df[df['Prediction'] == "High Risk"])
    low_risk = total - high_risk
    avg_interest = round(df["InterestRate"].astype(float).mean(), 2)
    avg_dti = round(df["DTIRatio"].astype(float).mean(), 2)

    import plotly.express as px
    pie_fig = px.pie(df, names="Prediction", title="Risk Distribution")
    pie_html = pie_fig.to_html(full_html=False)

    return render_template("dashboard.html",
                           total=total,
                           high_risk=high_risk,
                           low_risk=low_risk,
                           avg_interest=avg_interest,
                           avg_dti=avg_dti,
                           pie_chart=Markup(pie_html))

@app.route('/download-report', methods=['POST'])
def download_report():
    context = request.form.to_dict()
    pdf = render_pdf('pdf_report.html', context)
    if pdf:
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=loan_report.pdf'
        return response
    return "Error generating PDF", 500

if __name__ == '__main__':
    app.run(debug=True)