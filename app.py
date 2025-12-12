from flask import Flask, request, jsonify
import joblib
import numpy as np
from utils import make_feature_vector

MODEL_PATH = 'cybersecurity_model.pkl'
app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print('Warning: model not loaded. Run train_model.py first. Error:', e)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Accept either full feature dict or single 'features' list
    if 'features' in data:
        features = np.array(data['features']).reshape(1, -1)
    else:
        features = make_feature_vector(data)
    if model is None:
        return jsonify({'error': 'Model not loaded. Train model first.'}), 500
    pred = model.predict(features)
    prob = model.predict_proba(features).max(axis=1)[0] if hasattr(model, 'predict_proba') else None
    return jsonify({'Threat_Detected': bool(int(pred[0])), 'score': float(prob) if prob is not None else None})

@app.route('/')
def index():
    return 'AI-Powered Cybersecurity Threat Detection - API is running'

if __name__ == '__main__':
    app.run(debug=True)