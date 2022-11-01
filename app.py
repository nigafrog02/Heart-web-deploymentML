from contextlib import _RedirectStream
from pyexpat import features
from flask import redirect
from flask import Flask, render_template ,request,jsonify
import pickle
import numpy as np


#create flask app
app = Flask(__name__)
bag_model = pickle.load(open('heart.pkl','rb'))


@app.route('/')
def Home():
    return render_template('template.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = bag_model.predict(features)

    Age = request.form.get("Age")
    Gender = request.form.get("gender")
    Chestpain = request.form.get("chestpaintype")
    Chol = request.form.get("cholesterol")
    Thalach = request.form.get("Thalach")
    Exang = request.form.get("Exang")
    Oldpeak = request.form.get("Oldpeak")
    Slope = request.form.get("Slope")
    Ca = request.form.get("ca")
    Thal = request.form.get("Thal")


    return render_template("after.html" ,prediction = prediction,
    Gender = Gender, 
    Chestpain = Chestpain, 
    Exang = Exang,
    Slope = Slope,
    Ca = Ca,
    Thal = Thal,
    prediction_text = "Heart disease level {}".format(prediction),




    age = "Age: {}".format(Age), 
    chol="Cholesterol Level: {}".format(Chol),
    thalach="Thalach Value: {}".format(Thalach),
    oldpeak = "Oldpeak Value: {}".format(Oldpeak))


@app.route('/back', methods=['POST'])
def back():

    return redirect("http://127.0.0.1:8000/doctor/dochome", code=302)
    


if __name__ == "__main__":
    app.run(debug=True)
