#importing modules
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

plt.style.use('bmh')

# READING THE DATASETS
filepath = r"C:\Users\Namratha\Documents\Informac"
os.chdir(filepath)

inspection_data = pd.read_csv(filepath + "/LOS_ANGELES_COUNTY_RESTAURANT_AND_MARKET_INSPECTIONS.csv")
violation_data = pd.read_csv(filepath + "/LOS_ANGELES_COUNTY_RESTAURANT_AND_MARKET_VIOLATIONS.csv")


# ************************* INSPECTION DATA *******************************

#print(inspection_data.dtypes)
#df = inspection_data.describe(include = "all")
#df.to_excel("a.xlsx")

# splitting PE decription columns
def parse_description(str):
    reg = re.compile('.+(?=\()')
    desc = reg.search(str)
    if desc:
        return desc.group(0)

def parse_seating(str):
    reg = re.compile('(?<=\().+(?=\))')
    size = reg.search(str)
    if size:
        return size.group(0)

def parse_risk(str):
    return('  ').join(str.split(' ')[-2:])

inspection_data['type'] = inspection_data['PE DESCRIPTION'].apply(parse_description)
inspection_data['seating size'] = inspection_data['PE DESCRIPTION'].apply(parse_seating)
inspection_data['risk factor'] = inspection_data['PE DESCRIPTION'].apply(parse_risk)

# Deleting unnecessary Columns
cols = ['OWNER NAME', 'FACILITY NAME','PROGRAM NAME','PROGRAM ELEMENT (PE)', 'FACILITY ADDRESS','FACILITY STATE','SERVICE DESCRIPTION', 'PE DESCRIPTION']
#inspection_data.drop(cols, axis=1, inplace = True)

# Handling Categorical variables
inspection_data['ACTIVITY DATE'] = pd.to_datetime(inspection_data['ACTIVITY DATE'], format="%m/%d/%Y")

inspection_data['FACILITY ZIP'] = np.where(inspection_data['FACILITY ZIP'].str.len() == 5,
                                   inspection_data['FACILITY ZIP'],
                                   inspection_data['FACILITY ZIP'].str[:5])

#inspection_data['Zip Codes'] = inspection_data['Zip Codes'].astype('str')

inspection_data['SERVICE CODE'] = inspection_data['SERVICE CODE'].replace(401,2)

map = {'GRADE' : { 'A' : 1, 'B' : 2, 'C':3}}
inspection_data['GRADE'].replace(map, inplace = True)

# Deleting null values
inspection_data.dropna(how = "any", axis = 0, inplace=True)

#print(inspection_data['FACILITY ID'].value_counts())

#print(inspection_data.dtypes)
#print(inspection_data.isna().sum())

facility_score = pd.DataFrame(inspection_data.groupby(['PE DESCRIPTION'])['SCORE'].mean())
facility_score.reset_index(inplace=True)
facility_score = facility_score.sort_values(by='PE DESCRIPTION')
facility_score['SCORE'] = np.round(facility_score['SCORE'], 1)

ax = sns.barplot(y='PE DESCRIPTION', x='SCORE', data=facility_score, color='#081d58')
ax.set_xlabel('Average Score')
ax.set_ylabel('Facility type')
ax.set_xlim(facility_score.SCORE.min()-.5, facility_score.SCORE.max()+.5)
ax.set_title('Avg Score per Facility type')
sns.despine();

plt.show()

# ******************* VIOLATION DATA *******************

#print(violation_data.dtypes)
#df1 = violation_data.describe(include = "all")
#df1.to_excel("b.xlsx")

# Dropping unnecessary variables]
data = violation_data['VIOLATION  STATUS'].str.contains('OUT OF COMPLIANCE', na = False)
violation_data = violation_data[data]

violation_desc=violation_data.groupby(['VIOLATION DESCRIPTION','VIOLATION CODE']).size()
#print(pd.DataFrame({'Count':violation_desc.values}, index=violation_desc.index).sort_values(by = 'Count', ascending=False))

cols = ['VIOLATION  STATUS', 'VIOLATION DESCRIPTION', 'POINTS']
violation_data.drop(cols, axis=1, inplace = True)

#violation_data['VIOLATION  STATUS'] = violation_data['VIOLATION  STATUS'].astype('str')
#print(violation_data.dtypes)
#violation_data['VIOLATION  STATUS'] = violation_data.query('OUT OF COMPLIANCE' not in 'VIOLATION  STATUS')

#searchfor = ['john', 'doe']
#df = violation_data[violation_data['VIOLATION  STATUS'].str.contains()]

# Handling categorical variables
#violation_data['VIOLATION CODE'] = violation_data['VIOLATION CODE'].astype('category')

violation_data = pd.get_dummies(violation_data, columns = ['VIOLATION CODE'])
violation_data = violation_data.groupby('SERIAL NUMBER').sum().reset_index()
#print(violation_data.info())
#print(violation_data["VIOLATION  STATUS"].value_counts())
#print(violation_data["VIOLATION CODE"].value_counts())

# MERGING DATASETS
la_county_data = pd.merge(inspection_data, violation_data, on='SERIAL NUMBER', how='left')
la_county_data.dropna(how = "any", axis = 0, inplace = True)


count = np.where(la_county_data['FACILITY CITY']=='AVALON')
print(count['VIOLATION CODE'].value_counts())
#la_county_data.to_csv("final.csv")

#print(la_county_data["PROGRAM ELEMENT (PE)"].value_counts())
#print(la_county_data["FACILITY ZIP"].value_counts())
#print(la_county_data["GRADE"].value_counts())
#print(la_county_data["2011 Supervisorial District Boundaries (Official)"].value_counts())
#print(la_county_data["VIOLATION  STATUS"].value_counts())
#print(la_county_data["VIOLATION CODE"].value_counts())
#print(la_county_data.groupby(["VIOLATION  STATUS", "VIOLATION CODE"]).size())


grade_distribution = inspection_data.groupby('GRADE').size()
#print(pd.DataFrame({'Count of restaurant grade totals':grade_distribution.values}, index=grade_distribution.index))

cols = ['PROGRAM STATUS', 'SERVICE CODE', 'SCORE', 'GRADE', '2011 Supervisorial District Boundaries (Official)', 'Census Tracts 2010',
        'Board Approved Statistical Areas', 'risk factor', 'type']

inspection_num = inspection_data[cols]
#print(list(la_num.columns))

#inspection_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
#plt.show()









