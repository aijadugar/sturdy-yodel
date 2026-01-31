import numpy as np
import joblib

def preprocess_input(data, scaler, le_dict):
    # data: dict with keys matching feature names
    features = []

    for col in le_dict.keys():
        le = le_dict[col]
        val = data[col]
        features.append(le.transform([val])[0])

    # Add numeric features
    for col in ["Age", "Cough", "WeightLoss"]:
        features.append(data[col])

    # Scale numeric features
    features_array = np.array([features])
    features_array[:, -3:] = scaler.transform(features_array[:, -3:])
    return features_array