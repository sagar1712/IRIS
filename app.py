import numpy as np
from flask import Flask, request, render_template, url_for
from flask_material import Material
import pickle
import pandas as pd
import joblib

app = Flask(__name__)
model = pickle.load(open('data/model.pkl', 'rb'))
Material(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/preview')
def preview():
    df = pd.read_csv("data/Iris.csv")
    return render_template("preview.html", df_view=df)

species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
@app.route('/', methods=["POST"])
def analyze():
    if request.method == 'POST':
        petal_length = request.form['petal-length']
        sepal_length = request.form['sepal-length']
        petal_width = request.form['petal-width']
        sepal_width = request.form['sepal-width']

        sample_data = [sepal_length, sepal_width, petal_length, petal_width]
        clean_data = [float(i) for i in sample_data]

        # Reshape the Data as a Sample not Individual Features
        ex1 = np.array(clean_data).reshape(1, -1)
    model = joblib.load('data/model.pkl')
    result_prediction = model.predict(ex1)
	
    return render_template('index.html', petal_width=petal_width,
                           sepal_width=sepal_width,
                           sepal_length=sepal_length,
                           petal_length=petal_length,
                           clean_data=clean_data,
                           result_prediction=species[int(result_prediction)])

if __name__ == "__main__":
    app.run(debug=True)