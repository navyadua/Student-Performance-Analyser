from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
ml_model = pickle.load(open("model.pkl", "rb"))


#@app.route('/test')
#def test():
#    return 'Flask is being used for Development'


clusters = {0:'Good',1:'Excellent',2:'Can be Improved',3:'Sufficient',4:'Satisfactory'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            paid = int(request.form['q1'])
            internet = int(request.form['q2'])
            address = int(request.form['q3'])
            studytime = int(request.form['q4'])
            traveltime = int(request.form['q5'])
            walc = int(request.form['q6'])
            health = int(request.form['q7'])
            average = int(request.form['q8'])
#corr_columns = ['paid' , 'internet' , 'address',  'studytime', 'traveltime', 'Walc' ,'health','average']

            pred_args = [paid , internet , address,  studytime, traveltime, walc,health,average]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            print(pred_args_arr.reshape(1, -1))
            #mul_reg = open("multiple_linear_model.pkl", "rb")
            #ml_model = joblib.load(mul_reg)
            model_prediction = ml_model.predict(pred_args_arr)
            print(ml_model.predict(pred_args_arr))
            model_prediction = round(float(model_prediction))
            print(round(float(model_prediction)))
            predicted_cluster = clusters.get(model_prediction)
            print(predicted_cluster)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('prediction.html', prediction = predicted_cluster)

if __name__ == '__main__':
    app.run()