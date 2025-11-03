from flask import Flask, request, jsonify
from model_predictor import predict_species

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Iris Flower Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']  # Example: [5.1, 3.5, 1.4, 0.2]
    prediction = predict_species(features)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
