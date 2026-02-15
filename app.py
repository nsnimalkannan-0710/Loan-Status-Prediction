from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from pathlib import Path
import os

app = Flask(__name__)

# Model paths
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATHS = {
    'classification': BASE_DIR / 'models/trained_models/classification_model.pkl',
    'clustering': BASE_DIR / 'models/trained_models/clustering_model.pkl',
    'regression': BASE_DIR / 'models/trained_models/regression_model.pkl'
}

# Load models with error handling
models = {}
for name, path in MODEL_PATHS.items():
    try:
        models[name] = joblib.load(path)
        print(f"Successfully loaded {name} model")
    except Exception as e:
        print(f"Error loading {name} model: {str(e)}")
        models[name] = None

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'tab' not in data:
            return jsonify({'error': 'Invalid request'}), 400

        tab = data['tab']
        if tab not in models or models[tab] is None:
            return jsonify({'error': f'{tab.capitalize()} model not available'}), 503

        if tab == 'classification':
            # Prepare features for classification
            features = np.array([
                int(data.get('no_of_dependents', 0)),
                1 if data.get('education') == 'Graduate' else 0,
                1 if data.get('self_employed') == 'Yes' else 0,
                float(data.get('income_annum', 0)),
                float(data.get('loan_amount', 0)),
                int(data.get('loan_term', 1)),
                int(data.get('cibil_score', 300)),
                float(data.get('residential_assets_value', 0)),
                float(data.get('commercial_assets_value', 0)),
                float(data.get('luxury_assets_value', 0))
            ]).reshape(1, -1)

            prediction = models[tab].predict(features)[0]
            confidence = models[tab].predict_proba(features)[0][1] * 100 if hasattr(models[tab], 'predict_proba') else 85

            return jsonify({
                'status': 'Approved' if prediction else 'Rejected',
                'confidence': round(confidence, 2)
            })

        elif tab == 'clustering':
            # Prepare features for clustering
            features = np.array([
                float(data.get('income_annum', 0)),
                float(data.get('loan_amount', 0)),
                int(data.get('cibil_score', 300)),
                float(data.get('residential_assets_value', 0)),
                float(data.get('commercial_assets_value', 0)),
                float(data.get('luxury_assets_value', 0))
            ]).reshape(1, -1)

            segment_num = models[tab].predict(features)[0]
            segments = ['Standard', 'Preferred', 'Premium']  # Adjust based on your clustering model
            return jsonify({'segment': segments[min(segment_num, len(segments)-1)]})

        elif tab == 'regression':
            # Prepare features for regression
            features = np.array([
                int(data.get('no_of_dependents', 0)),
                1 if data.get('education') == 'Graduate' else 0,
                1 if data.get('self_employed') == 'Yes' else 0,
                float(data.get('income_annum', 0)),
                int(data.get('loan_term', 1)),
                int(data.get('cibil_score', 300)),
                float(data.get('residential_assets_value', 0)),
                float(data.get('commercial_assets_value', 0)),
                float(data.get('luxury_assets_value', 0))
            ]).reshape(1, -1)

            amount = models[tab].predict(features)[0]
            return jsonify({'amount': round(float(amount), 2)})

        return jsonify({'error': 'Invalid tab specified'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)