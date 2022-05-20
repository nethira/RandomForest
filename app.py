from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle
import json
import pandas as pd
from flask_cors import CORS, cross_origin

with open('shirt_length_model','rb') as f:
     mp=pickle.load(f)

with open('shirt_chest_width_model','rb') as f:
     mp1=pickle.load(f)

with open('shirt_waist_width_model','rb') as f:
     mp2=pickle.load(f)

with open('sleeve_length_model','rb') as f:
     mp3=pickle.load(f)

app=Flask(__name__)
# app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy dog'
# app.config['CORS_HEADERS'] = 'Content-Type'



@app.route('/')
# @cross_origin(origins='http://127.0.0.1:5500',headers=['Content-Type','Authorization'])
def index():
    return render_template('home.html')
    
@app.route('/predict', methods=['POST'])
# @cross_origin(origins='http://127.0.0.1:5500',headers=['Content-Type','Authorization'])
def home():
    print("data is:",request.data)
    print("datatype is:",type(request.data))
    my_json = request.data.decode('utf8').replace("'", '"')
    data = json.loads(my_json)
    print(data)
    print(type(data))
    data1 = data['age']
    data2 = data['shoe_size']
    data3 = data['height']
    data4 = data['weight']
    data5= data['body_athletic']
    data6= data['body_big']
    data7= data['body_regular']
    data8= data['body_slim']
    data9= data['fit_fitted']
    data10= data['fit_regular']
    print(data1,data2,data3,data4)
    datapoint = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9,data10]])
    pred_shirt_length={
         "name": "Shirt Length",
         "value": mp.predict(datapoint).tolist()[0],
         "unit": "cm"
    }
    print("type is" ,type(pred_shirt_length))
    pred_shirt_chest_width={
         "name": "Chest",
         "value": mp1.predict(datapoint).tolist()[0],
         "unit": "cm"
    }
    pred_shirt_waist_width={
         "name": "Waist",
         "value": mp2.predict(datapoint).tolist()[0],
         "unit": "cm"
    }
    pred_sleeve_length={
         "name": "Sleeve Length",
         "value": mp3.predict(datapoint).tolist()[0],
         "unit": "cm"
    }
    return jsonify({'shirt_length': pred_shirt_length, 'shirt_chest': pred_shirt_chest_width, 
    'shirt_waist': pred_shirt_waist_width,'shirt_sleeve':pred_sleeve_length})
#     return render_template('after.html', data1=pred_shirt_length,data2=pred_shirt_chest_width,data3=pred_shirt_waist_width,data4=pred_sleeve_length)



if __name__ == "__main__":
    app.run(debug=True)