import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained SVM model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "svm.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Features extracted from model specification
FEATURE_NAMES = [
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
    "exam_difficulty"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVR Model Predictor</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px 15px;
            color: #f8fafc;
        }

        .container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 800px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5),
                        0 0 30px rgba(99, 102, 241, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .container:hover {
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.6),
                        0 0 40px rgba(129, 140, 248, 0.3);
        }

        h1 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.2rem;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        p.subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 30px;
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .input-group label {
            margin-bottom: 8px;
            font-size: 0.85rem;
            text-transform: capitalize;
            color: #cbd5e1;
            letter-spacing: 0.5px;
        }

        .input-group input {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 12px 15px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        .input-group input:focus {
            border-color: #818cf8;
            box-shadow: 0 0 12px rgba(129, 140, 248, 0.4),
                        inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 15px;
            padding: 14px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);
            transition: all 0.3s ease;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 25px rgba(168, 85, 247, 0.5);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 12px;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            text-align: center;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
            animation: fadeIn 0.5s ease-in-out;
        }

        .result-box h2 {
            color: #34d399;
            font-size: 1.5rem;
            margin-top: 5px;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <h1>SVR Prediction System</h1>
    <p class="subtitle">Enter feature values below to generate model outcome</p>

    <form action="/predict" method="POST">
        <div class="form-grid">
            {% for feature in features %}
            <div class="input-group">
                <label for="{{ feature }}">{{ feature.replace('_', ' ') }}</label>
                <input type="number" step="any" name="{{ feature }}" id="{{ feature }}" required placeholder="0.0">
            </div>
            {% endfor %}
            <button type="submit" class="btn-submit">Predict Result</button>
        </div>
    </form>

    {% if prediction is not none %}
    <div class="result-box">
        <p style="color: #94a3b8; font-size: 0.9rem;">Predicted Output Value</p>
        <h2>{{ prediction }}</h2>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = [float(request.form[feat]) for feat in FEATURE_NAMES]
        features_array = np.array([input_data])
        prediction_val = model.predict(features_array)[0]
        formatted_prediction = round(prediction_val, 4)
    except Exception as e:
        formatted_prediction = f"Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES, prediction=formatted_prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
