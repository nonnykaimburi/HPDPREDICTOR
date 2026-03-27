import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

np.random.seed(42)

def generate_synthetic_longitudinal(n_patients=2500, seq_len=20):
    # Create patient-based synthetic data for longitudinal risk prediction
    patient_ids = np.arange(n_patients)
    age = np.random.normal(28, 5, n_patients)  # pregnant age
    bmi = np.random.normal(26, 4, n_patients)
    prior_hypertension = np.random.binomial(1, 0.18, n_patients)
    diabetes = np.random.binomial(1, 0.1, n_patients)

    # risk score baseline
    risk_base = 0.1 + 0.02 * (age - 25) + 0.05 * (bmi - 24) + 0.25 * prior_hypertension + 0.2 * diabetes
    risk_base = np.clip(risk_base, 0.02, 0.92)

    X_list = []
    y_class = []
    y_time = []

    for i in range(n_patients):
        rr = risk_base[i]
        event_happens = np.random.rand() < rr
        if event_happens:
            event_time = np.random.randint(5, seq_len - 1)
        else:
            event_time = seq_len + np.random.randint(1, 6)

        sbp = np.linspace(110, 118, seq_len) + np.random.normal(0, 5, seq_len)
        dbp = np.linspace(70, 76, seq_len) + np.random.normal(0, 3, seq_len)
        mapv = sbp * 0.4 + dbp * 0.6

        if event_happens:
            sbp[event_time:] += np.linspace(0, 30, seq_len - event_time)
            dbp[event_time:] += np.linspace(0, 20, seq_len - event_time)

        bmi_ts = bmi[i] + np.random.normal(0, 0.25, seq_len)
        weight_change = np.linspace(0, 4, seq_len) + np.random.normal(0, 0.5, seq_len)

        oliguria = np.random.binomial(1, 0.05 + 0.5*event_happens, seq_len)
        proteinuria = np.random.binomial(1, 0.03 + 0.6*event_happens, seq_len)

        extra = np.vstack([sbp, dbp, mapv, bmi_ts, weight_change, oliguria, proteinuria]).T
        X_list.append(extra)

        y_class.append(1 if event_happens else 0)
        y_time.append(event_time if event_happens else seq_len + 5)

    X = np.stack(X_list)
    y_class = np.array(y_class)
    y_time = np.array(y_time, dtype=np.float32)

    return X, y_class, y_time


def build_model(seq_len=20, n_features=7):
    # Use RandomForest for classification and regression
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    return clf, reg


def main():
    seq_len = 20
    X, y_class, y_time = generate_synthetic_longitudinal(n_patients=2500, seq_len=seq_len)

    # scale features per time step combined
    n_patients, T, n_features = X.shape
    X_2d = X.reshape(-1, n_features)
    scaler = StandardScaler().fit(X_2d)
    X_scaled = scaler.transform(X_2d).reshape(n_patients, T, n_features)

    # Flatten for sklearn
    X_flat = X_scaled.reshape(n_patients, -1)  # (n_patients, seq_len * n_features)

    X_trainval, X_test, y_class_trainval, y_class_test, y_time_trainval, y_time_test = train_test_split(
        X_flat, y_class, y_time, test_size=0.2, random_state=42, stratify=y_class)

    X_train, X_val, y_class_train, y_class_val, y_time_train, y_time_val = train_test_split(
        X_trainval, y_class_trainval, y_time_trainval, test_size=0.125, random_state=42, stratify=y_class_trainval)
    # 0.125 of 80% is 10% => 70/20/10

    clf, reg = build_model(seq_len=seq_len, n_features=n_features)
    print("Training classifier...")
    clf.fit(X_train, y_class_train)
    print("Training regressor...")
    reg.fit(X_train, y_time_train)

    # Evaluate
    from sklearn.metrics import accuracy_score, mean_squared_error
    pred_class = clf.predict(X_test)
    pred_time = reg.predict(X_test)
    acc = accuracy_score(y_class_test, pred_class)
    mse = mean_squared_error(y_time_test, pred_time)
    print(f'Test accuracy: {acc:.4f}')
    print(f'Test MSE: {mse:.4f}')

    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(clf, 'artifacts/risk_classifier.pkl')
    joblib.dump(reg, 'artifacts/time_regressor.pkl')
    np.save('artifacts/risk_scaler_mean.npy', scaler.mean_)
    np.save('artifacts/risk_scaler_scale.npy', scaler.scale_)

    # save train/val/test distribution info
    pd.DataFrame({'y_class': y_class_test, 'y_time': y_time_test}).to_csv('artifacts/test_labels.csv', index=False)
    print('Saved model and artifacts.')

if __name__ == '__main__':
    main()
