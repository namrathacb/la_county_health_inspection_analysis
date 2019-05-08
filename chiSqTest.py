import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats



# READING THE DATASETS
filepath = r"C:\Users\Namratha\Documents\Informac"
os.chdir(filepath)

inspection_data = pd.read_csv(filepath + "/health.csv")

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
data = inspection_data.select_dtypes(include=numerics)
data1 = pd.DataFrame(data)
cols = ['Latitude', 'Longitude', 'FACILITY ZIP', 'Zip Codes', 'month_old', 'year', 'Total Violations','month', 'Distance in miles', 'PROGRAM STATUS',
        'SERVICE CODE', '2011 Supervisorial District Boundaries  Official ','Census Tracts 2010','Board Approved Statistical Areas','type','risk factor', 'SCORE']
data1.drop(cols, axis = 1, inplace= True )

replace_map = {'GRADE': {'1' : 'A', '2':'B', '3':'C'}}
data1.replace(replace_map, inplace = True)

#data2 = data1.copy()
#data2.drop('GRADE', inplace= True, axis=1)

dummies = data1.idmax(axis=1)

print(dummies.head())

# Contingency table for Grade and Violation codes
#code_grade = pd.crosstab(data.GRADE, data.BORO, margins = True)
#boro_grade


