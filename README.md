🧠 Loan Default Risk Predictor

A smart, interactive web application built with Flask, Tailwind CSS, and Chart.js that predicts the risk of a loan default based on user input. The app visualizes risk levels, provides real-time tips to reduce risk, and ensures clean UX with session-based handling and auto-reset.


🚀 Features

✅ Predicts loan default probability using a trained ML model (scikit-learn)

📊 Displays risk and confidence using an animated Chart.js doughnut chart

💡 Shows personalized financial improvement tips

🧽 Form auto-resets after prediction cycle (modal → tips → reset)

🔒 Uses Flask sessions to avoid form re-submission issues

💅 Built with Tailwind CSS for modern responsive styling



📂 Project Structure

bash
Copy
Edit
loan-predictor/
│
├── app.py                 # Flask backend app
├── model.pkl              # Trained ML model
├── preprocessor.pkl       # Preprocessing pipeline
├── requirements.txt       # Python dependencies
├── Procfile               # Render.com deployment config
├── templates/
│   └── form.html          # Frontend UI with Tailwind + Chart.js


🧪 Local Setup

1. Clone the repo
bash
Copy
Edit
git clone https://github.com/MohiniVora/loan-predictor.git
cd loan-predictor

2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt

3. Run the app
bash
Copy
Edit
python app.py
Visit http://127.0.0.1:5000 in your browser.

🌍 Deployment on Render

Push this project to GitHub.

Go to Render.com.

Click "New Web Service", connect your repo.

Set:

Build Command: pip install -r requirements.txt

Start Command: python app.py or gunicorn app:app

Python Version: 3.11+

Environment Variables:

FLASK_ENV = production

SECRET_KEY = your_secure_key

📊 ML Model
Model Type: Binary classifier (predict_proba)

Output: Probability of default (Class 1)

Custom threshold: 0.5

Preprocessing: Stored in preprocessor.pkl (e.g., OneHotEncoder, StandardScaler, etc.)


📦 requirements.txt (example)

txt
Copy
Edit
flask
numpy
pandas
scikit-learn
joblib
gunicorn


🙌 Credits
ML Model: Built and trained using scikit-learn

UI: Tailwind CSS

Visualization: Chart.js

Deployment: Render.com

Check out the project on:

https://loan-default-risk.onrender.com
