import numpy as numpy
from flask import Flask, request, render_template
import pickle
import pandas as pd
import model

app = Flask(__name__, template_folder='templates', static_folder='static')
classifier = pickle.load(open('model/classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stroke_risk', methods=['POST'])
def stroke_risk():

    form_values = [float(x) for x in request.form.values()]
    features = pd.DataFrame({
    'gender': [form_values[0]],
    'age': [form_values[1]],
    'hypertension': [form_values[2]],
    'heart_disease': [form_values[3]],
    'ever_married': [form_values[4]],
    'work_type': [form_values[5]],
    'Residence_type': [form_values[6]],
    'avg_glucose_level': [form_values[7]],
    'bmi': [form_values[8]],
    'smoking_status': [form_values[9]]})
    prediction = classifier.predict(features)

    if prediction[0] == 1:
        return render_template('index.html', output='Your stroke risk is HIGH. Accuracy: ' + str(classifier.score(model.x_test, model.y_test) * 100) + '%')
    elif prediction [0] == 0:
        return render_template('index.html', output='Your stroke risk is low. Accuracy: ' + str(classifier.score(model.x_test, model.y_test) * 100) + '%')

    

if __name__ == '__main__':
    app.run()
