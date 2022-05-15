from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle
import json
import pandas as pd

with open('shirt_length_model','rb') as f:
     mp=pickle.load(f)

with open('shirt_chest_width_model','rb') as f:
     mp1=pickle.load(f)

with open('shirt_waist_width_model','rb') as f:
     mp2=pickle.load(f)

with open('sleeve_length_model','rb') as f:
     mp3=pickle.load(f)

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')
    
@app.route('/predict', methods=['POST'])
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
    data5=0
    data6=0
    data7=1
    data8=0
    data9=1
    data10=0
    print(data1,data2,data3,data4)
    datapoint = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9,data10]])
    pred_shirt_length=mp.predict(datapoint).tolist()
    print("type is" ,type(pred_shirt_length))
    pred_shirt_chest_width=mp1.predict(datapoint).tolist()
    pred_shirt_waist_width=mp2.predict(datapoint).tolist()
    pred_sleeve_length=mp3.predict(datapoint).tolist()
    return jsonify({'shirt_length': pred_shirt_length, 'shirt_chest': pred_shirt_chest_width, 'shirt_waist': pred_shirt_waist_width,'shirt_sleeve':pred_sleeve_length})
#     return render_template('after.html', data1=pred_shirt_length,data2=pred_shirt_chest_width,data3=pred_shirt_waist_width,data4=pred_sleeve_length)



if __name__ == "__main__":
    app.run(debug=True)