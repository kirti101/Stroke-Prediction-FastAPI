import pandas as pd
import numpy as np

from category_encoders import TargetEncoder

import pickle

def preprocess_data_test(df):

    #making copy of dataset

    df=df.copy()


    #dropping column 'id'

    #df=df.drop('id',axis=1)

 

    #binary encoding

    df['ever_married']=df['ever_married'].replace({'No':0,'Yes':1})

    df['Residence_type']=df['Residence_type'].replace({'Rural':0,'Urban':1})

   

    ##read pickle file of encoder

    pickle_in = open(r"target.pkl","rb")

    encoder=pickle.load(pickle_in)

    col_to_encode=['gender', 'work_type', 'smoking_status']

    df[col_to_encode] = encoder.transform(df[col_to_encode])

   

    #### Create a Pickle file using serialization

 

    #Handling Missing Values with Strategy Mean

    df['bmi'].fillna(df['bmi'].mean(), inplace=True )

 

    return df