from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import pickle as pc
import numpy as np
import joblib

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=["POST","GET"])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        try:
            area = float(request.form.get('area'))
            perimeter = float(request.form.get('perimeter'))
            major = float(request.form.get('major_axis_length'))
            minor = float(request.form.get('minor_axis_length'))
            ratio = float(request.form.get('aspect_ratio'))
            ecc = float(request.form.get('eccentricity'))
            convex = float(request.form.get('convex_area'))
            diameter = float(request.form.get('equiv_diameter'))
            extent = float(request.form.get('extent'))
            solidity = float(request.form.get('solidity'))
            roundess = float(request.form.get('roundess'))
            compactness = float(request.form.get('compactness'))
            shape1 = float(request.form.get('shape_factor_1'))
            shape2 = float(request.form.get('shape_factor_2'))
            shape3 = float(request.form.get('shape_factor_3'))
            shape4 = float(request.form.get('shape_factor_4'))

            data = [[area,perimeter,major, minor,ratio, ecc, convex,diameter, extent, solidity,roundess,compactness,shape1,shape2,shape3,shape4]]
            scaler = joblib.load('models/scaler_std.joblib')
            model = joblib.load('models/stacking_model.joblib')

            data_scaled = scaler.transform(data)
            predict = model.predict(data_scaled)
            predict = predict[0]
            return render_template('index.html', result = predict,area = area, perimeter = perimeter, major = major, minor = minor, ratio = ratio, ecc = ecc, convex = convex, diameter = diameter, extent = extent, solidity = solidity, roundess = roundess, compactness = compactness, shape1 = shape1, shape2 = shape2, shape3 = shape3, shape4 = shape4)
        except:
            err = 'Input Tidak Valid'
            return render_template('index.html', result=err)
        
if __name__ == '__main__':
	app.run(debug=True)