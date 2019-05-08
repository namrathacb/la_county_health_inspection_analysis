#importing modules
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('bmh')

# READING THE DATASETS
filepath = r"C:\Users\Namratha\Documents\Informac"
os.chdir(filepath)

inspection_data = pd.read_csv(filepath + "/a1.csv")

# ************************* INSPECTION DATA *******************************

# Deleting null values
inspection_data.dropna(how = "any", axis = 0, inplace=True)

# Correlation matrix

cols = ['PROGRAM STATUS', 'SERVICE DESCRIPTION', 'SCORE', 'GRADE', '2011 Supervisorial District Boundaries (Official)', 'Census Tracts 2010',
        'Board Approved Statistical Areas','risk factor', 'type']

inspection_num = inspection_data[cols]

inspection_num = pd.get_dummies(inspection_num, prefix='_',columns=['SERVICE DESCRIPTION'])
inspection_num = pd.get_dummies(inspection_num, prefix='grade', columns=['GRADE'])
inspection_num = pd.get_dummies(inspection_num, columns=['type'])
inspection_num = pd.get_dummies(inspection_num, columns=['risk factor'])
inspection_num = pd.get_dummies(inspection_num, prefix='program_', columns=['PROGRAM STATUS'])

f, ax = plt.subplots(figsize=(10, 10))
inspection_num_corr  = inspection_num.corr()
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            #square=True, ax=ax)


mask = np.zeros_like(inspection_num_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(inspection_num_corr, mask = mask, cmap='coolwarm',
             annot = True, vmin = -1)

#plt.show()


#top_violated_place = inspection_data["FACILITY ID","FACILITY NAME"].value_counts().head(15)
#print(pd.DataFrame({'Count':top_violated_place.values},index = top_violated_place.index))


#apply function to get only high, medium, low

temp = inspection_data.groupby('risk factor').size()
#print(temp.head())
#plot the histogram for the 3 levels of risk


temp1 = inspection_data[['FACILITY NAME','SCORE']].sort_values(['SCORE'],ascending = False).drop_duplicates()
#print(temp1.head(10))


temp1 = inspection_data[['FACILITY NAME','SCORE']].sort_values(['SCORE']).drop_duplicates()
print(temp1.head(10))

# Correlation matrix

cols = ['PROGRAM STATUS', 'SERVICE CODE', 'SCORE', 'GRADE', '2011 Supervisorial District Boundaries (Official)', 'Census Tracts 2010',
        'Board Approved Statistical Areas','risk factor', 'type']

inspection_num = inspection_data[cols]

inspection_num = pd.get_dummies(inspection_num, prefix='service_description_', columns=['SERVICE DESCRIPTION'])
inspection_num = pd.get_dummies(inspection_num, prefix='grade', columns=['GRADE'])
inspection_num = pd.get_dummies(inspection_num, prefix='type_', columns=['type'])
inspection_num = pd.get_dummies(inspection_num, prefix='risk_factor_', columns=['risk factor'])
inspection_num = pd.get_dummies(inspection_num, prefix='program_status_', columns=['PROGRAM STATUS'])

f, ax = plt.subplots(figsize=(10, 8))
corr = inspection_num.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
