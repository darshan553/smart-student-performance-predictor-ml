from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("student_performance_model.pkl")

@app.route("/")
def home():
    return "Student Performance Predictor API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        study_hours = data["study_hours"]
        previous_score = data["previous_score"]
        test_preparation = data["test_preparation"]

        features = np.array([[study_hours, previous_score, test_preparation]])
        prediction = model.predict(features)

        result = "Pass" if prediction[0] == 1 else "Fail"
        return jsonify({"prediction": result})

    except KeyError:
        return jsonify({"error": "Missing or incorrect input keys. Expected: study_hours, previous_score, test_preparation"}), 400

if __name__ == "__main__":
    app.run(debug=True)