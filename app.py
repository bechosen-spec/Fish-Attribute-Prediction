import json
import pickle

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the models
length_model = pickle.load(open('/home/boniface/Desktop/Projects/Fish-Attribute-Prediction/decision_tree_model.pkl', 'rb'))
weight_model = pickle.load(open('/home/boniface/Desktop/Projects/Fish-Attribute-Prediction/xgboost_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    data_array = np.array(list(data.values()), dtype=np.float32).reshape(1, -1)  # Reshape input data to a 2D array
    length = length_model.predict(data_array)[0]
    weight = weight_model.predict(data_array)[0]
    output = {'length': float(length), 'weight': float(weight)}  # Convert NumPy float32 to Python float
    return jsonify(output)

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    data_array = np.array(data).reshape(1, -1)  # Reshape input data to a 2D array
    length = length_model.predict(data_array)[0]
    weight = weight_model.predict(data_array)[0]
    return render_template("home.html", length=length, weight=weight)

if __name__ == "__main__":
    app.run(debug=True)
