from flask import Flask, render_template, request

import numpy as np

import pickle

app = Flask(__name__)

model = pickle.load(open("bmdmodel.pkl","rb"))

@app.route('/')

def index():

    return render_template("bmd.html")

@app.route('/predict', methods=['GET', 'POST'])

def predict():

    val1 = request.form['Age']

    val2 = request.form['gender']

    val3 = request.form['fracture']

    val4 = request.form['weight']

    val5=request.form['height']

    val6=request.form['medications']

    val7=request.form['medications']

    
    arr = np.array([val1, val2, val3, val4,val5,val6,val7])

    arr = arr.astype(np.float64)

    pred =model.predict([arr])

    return render_template("bmd.html", data = "Your Bone Mineral Density is {}" .format("".join(str(i) for i in pred)))


if __name__ == '__main__':

    app.run(debug=True)