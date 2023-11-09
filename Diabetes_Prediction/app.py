from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

application = Flask(__name__)
app = application


with open('D:\Machine Learning project\Diabetes_Prediction\models\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('D:\Machine Learning project\Diabetes_Prediction\models\classifier.pkl','rb') as file:
    classifier = pickle.load(file)

val = classifier.predict(scaler.transform([[1,85,66,29,30.5,26.6,0.351,31]]))
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ['POST','GET'])
def predict_datapoint():

    result = ""

    if request.method == 'POST':
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = int(request.form.get('Glucose'))
        BloodPressure = int(request.form.get('BloodPressure'))
        SkinThickness = int(request.form.get('SkinThickness'))
        Insulin   = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = int(request.form.get('Age'))

        predicted = classifier.predict(scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]))

        if predicted==1:
            result = "Diabetes"
        else:
            result = "Not-Diabetes"
        
        return render_template('results.html',result=result)
    else:
        return render_template('home.html') 

if __name__=="__main__":
    app.run(host='0.0.0.0')


