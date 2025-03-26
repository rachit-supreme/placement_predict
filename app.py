from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   
    float_features = [int(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)

    output =(prediction[0])

    return render_template('index.html', prediction_text='Placement status : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)