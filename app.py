from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model_saved', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods = ['POST'])
def predict():
    age = request.form.get('age')
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trestbps = request.form.get('trestbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')

    input_query = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    result = model.predict(input_query)[0]

    return jsonify({'Result' : result})

if __name__ == '__main__':
    app.run(debug=True)