# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:14:34 2022

@author: hp
"""

#pip install fastapi uvicorn

# 1. Library imports
import uvicorn
from fastapi import FastAPI, Form
from Stroke import Stroke
import numpy as np
import pickle
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from data_preprocess import *

# 2. Create the app object
app = FastAPI(title="Stroke Prediction API",
    description=" A simple API that predicts about whether a person can get a stroke or not.")


pickle_in = open("best_forest.pkl","rb")
classifier=pickle.load(pickle_in)

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome! click on predict to find stroke prediction': f'{name}'}

@app.post('/predict')
def predict_Stroke(data:Stroke):
    data = data.dict()
    df=pd.DataFrame.from_dict(data,orient='index').T
    #print(df)
    test_dataset=preprocess_data_test(df)
    test_dataset=test_dataset.to_numpy()
    print(test_dataset)
    prediction = classifier.predict(test_dataset)
    # if(prediction[0]==0):
    if(prediction[0]==0):
        prediction="Person does not has Stroke."
    else:
        prediction="Person has Stroke."
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
