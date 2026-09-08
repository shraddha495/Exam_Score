import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained SVR model
MODEL_PATH = "svm.pkl"
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# HTML Template with inline CSS styling, box-shadows, and interactive UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVR Model Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --card-bg: rgba(255, 255, 255, 0.95);
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --shadow-soft: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
            --shadow-hover: 0 20px 30px -10px rgba(79, 70, 229, 0.3);
            --shadow-inset: inset 0 2px 4px 0 rgba(0, 0, 0, 0.05);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            background: var(--card-bg);
            width: 100%;
            max-width: 850px;
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .container:hover {
            box-shadow: 0 25px 35px -5px rgba(0, 0, 0, 0.4);
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        header h1 {
            color: var(--text-main);
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 0.4rem;
            text-transform: capitalize;
        }

        .form-group input {
            padding: 0.75rem 1rem;
            border: 1.5px solid var(--border-color);
            border-radius: 10px;
            font-size: 0.95rem;
            outline: none;
            box-shadow: var(--shadow-inset);
            transition: all 0.2s ease-in-out;
            background: #f8fafc;
        }

        .form-group input:focus {
            border-color: var(--primary-color);
            background: #ffffff;
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.15);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 0.9rem 1.5rem;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
            transition: all 0.25s ease;
        }

        .btn-submit:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
            box-shadow: var(--shadow-hover);
            transform: translateY(-2px);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            animation: fadeIn 0.4s ease-out;
        }

        .result-card h2 {
            color: #166534;
            font-size: 1.1rem;
            margin-bottom: 0.3rem;
        }

        .result-card .prediction-value {
            color: #15803d;
            font-size: 2rem;
            font-weight: 700;
        }

        .error-card {
            margin-top: 2rem;
            padding: 1rem;
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
            border-radius: 10px;
            text-align: center;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>SVR Model Prediction</h1>
            <p>Enter feature parameters to generate model outputs</p>
        </header>

        <form action="/predict" method="POST" class="grid-form">
            {% for feature in features %}
            <div class="form-group">
                <label for="{{ feature }}">{{ feature.replace('_', ' ') }}</label>
                <input type="number" step="any" id="{{ feature }}" name="{{ feature }}" required placeholder="0.00">
            </div>
            {% endfor %}

            <button type="submit" class="btn-submit">Generate Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h2>Predicted Output Value</h2>
            <div class="prediction-value">{{ prediction }}</div>
        </div>
        {% endif %}

        {% if error %}
        <div class="error-card">
            {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Extracted input features from your SVM object metadata
FEATURES = [
    "age",
    "gender",
    "course",
    "study_hours",
    "class_attendance",
    "internet_access",
    "sleep_hours",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty",
]


@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_TEMPLATE, features=FEATURES, prediction=None, error=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURES,
            prediction=None,
            error="Model file 'svm.pkl' not loaded on server.",
        )

    try:
        # Collect and order features correctly
        input_data = [float(request.form[feature]) for feature in FEATURES]
        features_array = np.array([input_data])

        # Run prediction
        raw_prediction = model.predict(features_array)[0]
        formatted_prediction = round(float(raw_prediction), 4)

        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURES,
            prediction=formatted_prediction,
            error=None,
        )

    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURES,
            prediction=None,
            error=f"Prediction Error: {str(e)}",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
