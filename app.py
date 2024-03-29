from flask import Flask, request, render_template
import numpy as np
import pickle

model = pickle.load(open('logistic_reg_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    glucose = float(request.form.get('glucose'))
    bmi = float(request.form.get('bmi'))
    age = float(request.form.get('age'))
    
    data = np.array([[glucose, bmi, age]])
    data = data.astype(float)

    # print("================DATA===================", data)

    pred = model.predict(data)
    
    # print("================Prediction===================", pred)

    results = round(pred[0], 2)
    return render_template('index.html', results=results, pred_df=pred)

if __name__ == '__main__':
    app.run(port=3000, debug=True)