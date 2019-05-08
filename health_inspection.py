import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor


# READING THE DATASETS
filepath = r"C:\Users\Namratha\Documents\Informac"
os.chdir(filepath)

inspection_data = pd.read_csv(filepath + "/final.csv")
#print(inspection_data.dtypes)

replace_map = {'type': {'RESTAURANT': 1, 'FOOD MKT RETAIL': 2, 'INTERIM HOUSING FF': 3, 'LIC HTH CARE FOOD FAC': 4,
                                  'CATERER': 5, 'FOOD PROCESSING WHOLESALE': 6, 'FOOD VEHICLE COMMISSARY': 7 , 'FOOD WAREHOUSE': 8 ,
                        'MARKET  WHOLESALE ': 9,'FOOD  STAND': 10,'FOOD  COMPLEX': 11}}

inspection_data.replace(replace_map, inplace=True)

map = {'risk factor': {'HIGH RISK' : 1, 'MODERATE RISK' : 2, 'LOW RISK':3}}
inspection_data.replace(map, inplace=True)
map1 = {'PROGRAM STATUS': {'ACTIVE':1, 'INACTIVE':2}}
inspection_data.replace(map1, inplace=True)
map2 = {'GRADE' : { 'A' : 1, 'B' : 2, 'C':3}}
inspection_data['GRADE'].replace(map2, inplace = True)
#inspection_data['FACILITY CITY'] = inspection_data['FACILITY CITY'].astype('category').cat.codes()
#inspection_data['ACTIVITY DATE'] = pd.to_datetime(inspection_data['ACTIVITY DATE'], format="%m/%d/%Y")

column= ['OWNER ID', 'FACILITY ID', 'RECORD ID', 'FACILITY ZIP', 'SERIAL NUMBER', 'EMPLOYEE ID', 'Zip Codes']

#inspection_data[column] = inspection_data[column].astype('str')
#print(inspection_data.dtypes)

#print(inspection_data['type'].value_counts())
#print(inspection_data['seating size'].value_counts())
#print(inspection_data['risk factor'].value_counts())

# Deleting null values
inspection_data.dropna(how = "any", axis = 0, inplace=True)
#inspection_data.to_csv("finalData.csv")


#print(inspection_data.isna().sum())
#print(inspection_data.dtypes)


# NUMERICAL DATA DISTRIBUTION

cols = ['PROGRAM STATUS', 'SERVICE CODE', 'SCORE', 'GRADE', '2011 Supervisorial District Boundaries (Official)', 'Census Tracts 2010',
        'Board Approved Statistical Areas', 'risk factor', 'type']

inspection_num = inspection_data[cols]
#print(list(la_num.columns))

#inspection_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
#plt.show()


# CORRELATION ANALYSIS

inspection_num_corr = inspection_num.corr()
mask = np.zeros_like(inspection_num_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(inspection_num_corr, mask = mask, cmap='coolwarm',
             annot = True, vmin = -1)

#sns.heatmap(corr[1:,:-1], mask=mask[1:,:-1], cmap='inferno', vmin = -0.1, vmax=0.8, square=True)
# NON NUMERICAL DATA DISTRIBUTION

#la_cat = la_county_data.select_dtypes(include=['category']).copy()
#print(la_cat.columns)

#plt.show()


# Train and test

#print(inspection_num.dtypes)

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
data = inspection_data.select_dtypes(include=numerics)
data = pd.DataFrame(data)
y = data['SCORE']
data.drop('SCORE', axis = 1, inplace= True )


# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)

dt_clf = DecisionTreeRegressor(splitter="random", max_leaf_nodes=16, random_state=0)
bag_clf = BaggingRegressor(dt_clf, n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=0)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#iris = load_iris()
##no early stoping defined, so it goes the full length
rnd_clf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

for name, score in zip(X_train, rnd_clf.feature_importances_):
    print(name, score)


inspection_data['risk factor'] = inspection_data['risk factor'].astype('int64')
inspection_data['year'] = pd.DatetimeIndex(inspection_data['ACTIVITY DATE']).year
inspection_data['month'] = pd.DatetimeIndex(inspection_data['ACTIVITY DATE']).month

inspection_data.dropna(how = "any", axis = 0, inplace=True)

#year = pd.Categorical(inspection_data.year)
#inspection_data = inspection_data.set_index(['FACILITY ID', 'year'])

inspection_data.to_csv('health.csv')



