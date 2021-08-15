from flask import Flask, render_template, request ,redirect
import requests
import pickle
import numpy as np
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)
model=joblib.load("performance_predictor_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    try:
        inft=[float(x) for x in request.form.values()]
        value=np.array(inft)
    except ValueError as e:
        return render_template('index.html',flag=3)
    else:
        if inft[0]<0 or inft[0]>24:
            return render_template('index.html',flag=1)
        elif inft[0]>12:
            return render_template('index.html',flag=2)
        else:
            out=model.predict([value])[0][0].round(2)
            return render_template('index.html',hr=inft[0],flag=out)
if __name__ == '__main__':
    app.run(debug=True)