import pickle
import pandas as pd

with open('shirt_length_model','rb') as f:
     mp=pickle.load(f)

with open('shirt_chest_width_model','rb') as f:
     mp1=pickle.load(f)

with open('shirt_waist_width_model','rb') as f:
     mp2=pickle.load(f)

with open('sleeve_length_model','rb') as f:
     mp3=pickle.load(f)

datapoint = pd.DataFrame({"age": [30.0], "foot_length": [49.0], "height": [155],"weight":[67],"body_type_athletic":[0],"body_type_big":[0]	,"body_type_regular":[1],	"body_type_slim":[0],	"fit_preference_fitted":[1],	"fit_preference_regular":[0]})

pred_shirt_length=mp.predict(datapoint)
pred_shirt_chest_width=mp1.predict(datapoint)
pred_shirt_waist_width=mp2.predict(datapoint)
pred_sleeve_length=mp3.predict(datapoint)

print("shirt_length: ",pred_shirt_length.item())
print("shirt_chest_width: ",pred_shirt_chest_width.item())
print("shirt_waist_width: ",pred_shirt_waist_width.item())
print("sleeve_length_model: ",pred_sleeve_length.item())