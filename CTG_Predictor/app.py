from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('model/fetal_health_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            data = [
                float(request.form['prolongued_decelerations']),
                float(request.form['abnormal_short_term_variability']),
                float(request.form['percentage_of_time_with_abnormal_long_term_variability']),
                float(request.form['accelerations']),
                float(request.form['histogram_mode']),
                float(request.form['histogram_mean']),
                float(request.form['mean_value_of_long_term_variability']),
                float(request.form['histogram_variance']),
                float(request.form['histogram_median']),
                float(request.form['uterine_contractions'])
            ]

            features = np.array(data).reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction = classifier.predict(features_scaled)

            return jsonify({'fetal_health': int(prediction[0])})

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about_data():
    return render_template('DataANA.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
