import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import lasso regressor model and standard scaler pickle
lasso_model = pickle.load(open('lassomodel.pkl', 'rb'))
standard_scaler = pickle.load(open('scaling.pkl', 'rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Extracting data from form
        season = int(request.form.get('season'))
        yr = int(request.form.get('yr'))
        mnth = int(request.form.get('mnth'))
        weekday = int(request.form.get('weekday'))
        weathersit = int(request.form.get('weathersit'))
        atemp = float(request.form.get('atemp'))
        hum = float(request.form.get('hum'))
        windspeed = float(request.form.get('windspeed'))

        # Transforming input data using standard scaler
        new_data_scaled = standard_scaler.transform([[season, yr, mnth, weekday, weathersit, atemp, hum, windspeed]])

        # Making prediction
        result = lasso_model.predict(new_data_scaled)

        return render_template('home.html', result=round(result[0]))

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
