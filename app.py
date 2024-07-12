from flask import Flask, request, render_template
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

iris = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data, iris.target)

app = Flask(__name__)

@app.route('/')
def home():
    return 'Random forest ka server hega'

@app.route('/<name>')
def testingRoute(name):
    return f'hello there {name}'

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    species = iris.target_names[prediction[0]]
    return render_template('index.html', prediction_text=f'The Iris species is: {species}')

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5000)