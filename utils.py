from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np

SCALER_PATH = 'scaler.joblib'

def fit_scaler(X_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    joblib.dump(scaler, SCALER_PATH)
    return Xs

def transform_with_saved_scaler(X):
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found. Fit scaler first.")
    scaler = joblib.load(SCALER_PATH)
    return scaler.transform(X)

def make_feature_vector(payload):
    keys = ['packet_size','failed_logins','request_frequency','src_bytes','dst_bytes','duration']
    vec = [float(payload.get(k, 0.0)) for k in keys]
    return np.array(vec).reshape(1, -1)
