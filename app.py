import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify

# Load ML model
model = pickle.load(open('model_saved', 'rb'))

# Create application
app = Flask(__name__)

# Bind home function to URL
@app.route('/')
def home():
    return render_template('Heart Disease Classifier.html')

# Bind predict function to URL
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        json_data = request.get_json()

        # Convert JSON data to a list of numeric values
        features = [float(json_data[key]) for key in json_data]

        # Convert features to array
        array_features = [np.array(features)]

        # Predict features
        prediction = model.predict(array_features)

        # Additional information based on the prediction value
        result_text = 'The patient is not likely to have heart disease!' if prediction[
                                                                                0] == 1 else 'The patient is likely to have heart disease!'

        # Return the prediction and additional information as JSON
        return jsonify({"Result": int(prediction[0]), "ResultText": result_text})


    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Run the application
    app.run()
