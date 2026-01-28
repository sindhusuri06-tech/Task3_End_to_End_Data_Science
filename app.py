from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("student_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Student Performance Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    study_hours = np.array([[data['study_hours']]])
    prediction = model.predict(study_hours)
    return jsonify({'predicted_marks': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
