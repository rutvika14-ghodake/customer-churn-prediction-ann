import os
import sys
import gc

# 1. ENVIRONMENT VARIABLES & MEMORY LIMITS (MUST RUN BEFORE TF IS IMPORTED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'         
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"         

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

# 2. IMPORT TENSORFLOW & APPLY PARALLELISM CONFIGS IMMEDIATELY
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

gc.enable()

app = Flask(__name__)

# 3. LOAD PREPROCESSING PIPELINES & MODEL
with open('categorical_encoder.pkl', 'rb') as f:
    transformer = joblib.load(f)

with open('numerical_scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

model = tf.keras.models.load_model('ann_model.keras')
gc.collect()

# 4. EMBEDDED HTML TEMPLATE
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Customer Churn Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #f4f7f6 0%, #e9ecef 100%);
            --card-bg: #ffffff;
            --primary: #2b5c8f;
            --primary-hover: #1e4366;
            --text-main: #333333;
            --text-muted: #666666;
            --border-color: #dddddd;
            --risk-high: #dc3545;
            --risk-low: #28a745;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: var(--bg-gradient); color: var(--text-main); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; }
        .container { width: 100%; max-width: 950px; background: var(--card-bg); border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); overflow: hidden; display: grid; grid-template-columns: 1fr; }
        @media (min-width: 768px) { .container { grid-template-columns: 1.2fr 0.8fr; } }
        .form-section { padding: 40px; }
        .results-section { background: #f8fafc; padding: 40px; border-left: 1px solid #edf2f7; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
        h2 { font-size: 24px; font-weight: 700; color: var(--primary); margin-bottom: 8px; }
        .subtitle { font-size: 14px; color: var(--text-muted); margin-bottom: 30px; }
        .grid-inputs { display: grid; grid-template-columns: 1fr; gap: 18px; }
        @media (min-width: 480px) { .grid-inputs { grid-template-columns: 1fr 1fr; } }
        .input-group { display: flex; flex-direction: column; }
        label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; color: var(--text-main); }
        input, select { padding: 10px 14px; font-size: 14px; border: 1px solid var(--border-color); border-radius: 8px; background: #fafafa; color: var(--text-main); transition: all 0.2s ease; width: 100%; }
        input:focus, select:focus { outline: none; border-color: var(--primary); background: #ffffff; box-shadow: 0 0 0 3px rgba(43, 92, 143, 0.1); }
        .submit-btn { grid-column: 1 / -1; background: var(--primary); color: white; border: none; padding: 14px; font-size: 16px; font-weight: 600; border-radius: 8px; cursor: pointer; transition: background 0.2s ease; margin-top: 10px; }
        .submit-btn:hover { background: var(--primary-hover); }
        .result-card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.04); width: 100%; max-width: 320px; }
        .gauge-text { font-size: 48px; font-weight: 700; margin: 15px 0; }
        .high-risk { color: var(--risk-high); }
        .low-risk { color: var(--risk-low); }
        .badge { display: inline-block; padding: 6px 12px; font-size: 12px; font-weight: 700; border-radius: 50px; text-transform: uppercase; }
        .badge-danger { background: #fde8e8; color: var(--risk-high); }
        .badge-success { background: #def7ec; color: var(--risk-low); }
        .placeholder-text { color: var(--text-muted); font-size: 15px; line-height: 1.5; }
    </style>
</head>
<body>
<div class="container">
    <div class="form-section">
        <h2>Customer Metrics</h2>
        <p class="subtitle">Enter bank client characteristics below to run prediction matrix</p>
        <form action="/predict" method="POST" class="grid-inputs">
            <div class="input-group"><label>Credit Score</label><input type="number" name="CreditScore" min="300" max="850" value="{{ original_inputs.CreditScore if original_inputs else '600' }}" required></div>
            <div class="input-group"><label>Geography</label><select name="Geography" required><option value="France" {{ 'selected' if original_inputs and original_inputs.Geography == 'France' }}>France</option><option value="Germany" {{ 'selected' if original_inputs and original_inputs.Geography == 'Germany' }}>Germany</option><option value="Spain" {{ 'selected' if original_inputs and original_inputs.Geography == 'Spain' }}>Spain</option></select></div>
            <div class="input-group"><label>Gender</label><select name="Gender" required><option value="Female" {{ 'selected' if original_inputs and original_inputs.Gender == 'Female' }}>Female</option><option value="Male" {{ 'selected' if original_inputs and original_inputs.Gender == 'Male' }}>Male</option></select></div>
            <div class="input-group"><label>Age</label><input type="number" name="Age" min="18" max="100" value="{{ original_inputs.Age if original_inputs else '40' }}" required></div>
            <div class="input-group"><label>Tenure (Years)</label><input type="number" name="Tenure" min="0" max="10" value="{{ original_inputs.Tenure if original_inputs else '5' }}" required></div>
            <div class="input-group"><label>Account Balance ($)</label><input type="number" step="0.01" name="Balance" value="{{ original_inputs.Balance if original_inputs else '60000.00' }}" required></div>
            <div class="input-group"><label>Number of Products</label><input type="number" name="NumOfProducts" min="1" max="4" value="{{ original_inputs.NumOfProducts if original_inputs else '1' }}" required></div>
            <div class="input-group"><label>Has Credit Card?</label><select name="HasCrCard" required><option value="1" {{ 'selected' if original_inputs and original_inputs.HasCrCard == '1' }}>Yes (1)</option><option value="0" {{ 'selected' if original_inputs and original_inputs.HasCrCard == '0' }}>No (0)</option></select></div>
            <div class="input-group"><label>Is Active Member?</label><select name="IsActiveMember" required><option value="1" {{ 'selected' if original_inputs and original_inputs.IsActiveMember == '1' }}>Yes (1)</option><option value="0" {{ 'selected' if original_inputs and original_inputs.IsActiveMember == '0' }}>No (0)</option></select></div>
            <div class="input-group"><label>Estimated Salary ($)</label><input type="number" step="0.01" name="EstimatedSalary" value="{{ original_inputs.EstimatedSalary if original_inputs else '50000.00' }}" required></div>
            <button type="submit" class="submit-btn">Run ANN Analytics</button>
        </form>
    </div>
    <div class="results-section">
        {% if prediction_text %}
            <div class="result-card">
                <h3>{{ prediction_text }}</h3>
                <div class="gauge-text {{ 'high-risk' if will_leave else 'low-risk' }}">{{ churn_risk }}%</div>
                <div style="margin-bottom: 15px;"><p style="font-size: 13px; color: var(--text-muted);">Probability of Attrition</p></div>
                {% if will_leave %}<span class="badge badge-danger">High Churn Risk</span>{% else %}<span class="badge badge-success">Stable Customer</span>{% endif %}
            </div>
        {% elif error_text %}
            <div class="result-card" style="border-top: 4px solid var(--risk-high)"><p class="high-risk" style="font-weight: 600;">System Error</p><p style="font-size: 13px; margin-top: 5px;">{{ error_text }}</p></div>
        {% else %}
            <div class="placeholder-text"><p>Fill out the customer metrics and click <strong>Run ANN Analytics</strong> to evaluate churn status.</p></div>
        {% endif %}
    </div>
</div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender_map = {'Female': 0, 'Male': 1}
        gender_encoded = gender_map.get(request.form['Gender'], 0)

        data = {
            'CreditScore': [float(request.form['CreditScore'])],
            'Geography': [request.form['Geography']],
            'Gender': [gender_encoded],
            'Age': [float(request.form['Age'])],
            'Tenure': [float(request.form['Tenure'])],
            'Balance': [float(request.form['Balance'])],
            'NumOfProducts': [float(request.form['NumOfProducts'])],
            'HasCrCard': [float(request.form['HasCrCard'])],
            'IsActiveMember': [float(request.form['IsActiveMember'])],
            'EstimatedSalary': [float(request.form['EstimatedSalary'])]
        }
        input_df = pd.DataFrame(data)

        if hasattr(transformer, 'transform'):
            features = transformer.transform(input_df)
        else:
            features = input_df

        if hasattr(scaler, 'transform'):
            final_features = scaler.transform(features)
        else:
            final_features = features

        # Lightweight Direct Model Invocation
        input_data = np.asarray(final_features, dtype=np.float32)
        prediction_prob = model(input_data, training=False).numpy()

        churn_risk = round(float(prediction_prob[0][0]) * 100, 2)
        will_leave = churn_risk >= 50.0

        gc.collect()

        return render_template_string(
            HTML_TEMPLATE,
            prediction_text="Churn Risk Analysis Complete",
            churn_risk=churn_risk,
            will_leave=will_leave,
            original_inputs=request.form
        )
    except Exception as e:
        gc.collect()
        return render_template_string(HTML_TEMPLATE, error_text=f"Prediction Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
