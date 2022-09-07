# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:47:35 2022

@author: hp
"""

from pydantic import BaseModel
from typing import Optional

class Stroke(BaseModel):
    gender: object
    age: float
    hypertension: int 
    heart_disease: int 
    ever_married: object
    work_type: object
    Residence_type: object
    avg_glucose_level: Optional[float] = 0.0
    bmi: Optional[float] = 0.0
    smoking_status: object

    