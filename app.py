from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
#loading the model
model = pickle.load(open('heartDisease_model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():    
    """Grabs the input values and uses them to make prediction"""
    age = int(request.form["age"])
    sexe = int(request.form["sex"])
    cpt = int(request.form["cpt"])
    bp = int(request.form["bp"])
    chol = int(request.form["chol"])
    fbs = int(request.form["fbs"])
    ecg = int(request.form["ecg"])
    hr = int(request.form["hr"])
    exang = int(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = int(request.form["slope"])
    majVessel = int(request.form["majVessel"])
    thal = int(request.form["thal"])

    #  Convert all the features into a numpy array
    features = np.array([age, sexe, cpt, bp, chol, fbs, ecg, hr, exang, oldpeak, slope, majVessel,thal])
    features = np.reshape(features,(1, features.shape[0]))
    prediction = model.predict(features)
    value = prediction[0]

    if value == 0:
        return render_template('index.html', prediction_result = f'No heart disease. You got an healthy heart')
    else:
        return render_template('index.html', prediction_result = f'Heart failure possibility. You have to take a good care of your health.')


if __name__ == "__main__":
    app.run()