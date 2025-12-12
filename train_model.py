import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

SCALER_PATH = 'scaler.joblib'

def main(data_path, out_model):
    df = pd.read_csv(data_path)
    if 'Attack_Label' not in df.columns:
        raise ValueError('Dataset must include Attack_Label column')

    X = df.drop(columns=['Attack_Label'])
    y = df['Attack_Label']

    # Basic checks
    print("Data shape:", X.shape)
    print("Label distribution:\n", y.value_counts())

    # split first (no preprocessing before split)
    stratify_arg = y if y.nunique() > 1 and min(y.value_counts()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    # Fit scaler only on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Evaluate on test set (this is the proper test)
    y_pred = model.predict(X_test_scaled)
    try:
        y_proba = model.predict_proba(X_test_scaled)[:,1]
        roc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc = None

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    if roc is not None:
        print("ROC AUC:", roc)

    # Optional: cross-validation on training set
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print("CV accuracy (train):", np.mean(cv_scores), "±", np.std(cv_scores))

    # Save model
    joblib.dump(model, out_model)
    print(f"Model saved to {out_model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='cybersecurity_model.pkl')
    args = parser.parse_args()
    main(args.data, args.out)