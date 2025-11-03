import joblib
import numpy as np

def predict_species(features, model_path="../models/iris_model.joblib"):
    """Predict species name from feature input."""
    model = joblib.load(model_path)
    prediction = model.predict([features])
    species = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return species[int(prediction)]
