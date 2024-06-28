from flask import Flask, jsonify, render_template, request
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle as pc
import numpy as np

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['GET', 'POST'])

def dashboard():
    if request.method == 'GET':
        return render_template('dashboard.html')
    elif request.method == 'POST':
        try:
            area = float(request.form.get('area'))
            major = float(request.form.get('major'))
            minor = float(request.form.get('minor'))
            ecc = float(request.form.get('ecc'))
            convex = float(request.form.get('convex'))
            extent = float(request.form.get('extent'))
            perimeter = float(request.form.get('perimeter'))

            data = np.array([[area, major, minor, ecc, convex, extent, perimeter]])
            pre_processing = pc.load(open('models/my_preprocessing.pickle', 'rb'))
            predict_model = pc.load(open('models/bagging_model.pickle', 'rb'))

            data_encode = pre_processing.transform(data)
            prediction = predict_model.predict(data_encode)
            prediction = prediction[0]

            return render_template('dashboard.html', result=prediction, area=area, major=major,
                                   minor=minor, ecc=ecc, convex=convex, extent=extent,
                                   perimeter=perimeter)
        except:
            err = 'Inputan anda invalid input hanya berupa angka'

        return render_template('dashboard.html', result=err)

if __name__ == '__main__':
    app.run(debug=True)