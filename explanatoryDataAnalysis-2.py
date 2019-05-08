import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# READING THE DATASETS
filepath = r"C:\Users\Namratha\Documents\Informac"
os.chdir(filepath)

inspection_data = pd.read_csv(filepath + "/health.csv")

#apply function to get only high, medium, low

cols = ['PROGRAM STATUS', 'SERVICE CODE', 'SCORE', 'GRADE', '2011 Supervisorial District Boundaries  Official ', 'Census Tracts 2010',
        'Board Approved Statistical Areas', 'risk factor', 'type', 'Distance in miles']

inspection_num = inspection_data[cols]
#print(list(la_num.columns))

inspection_num_corr = inspection_num.corr()
mask = np.zeros_like(inspection_num_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(inspection_num_corr, mask = mask, cmap='coolwarm',
             annot = True, vmin = -1)

plt.show()

violation_desc2 = inspection_data.groupby(['FACILITY ID']).size()
print(pd.DataFrame({'Total Violations':violation_desc2['Total Violations']}, index=violation_desc2.index).sort_values(by='Total Violations', ascending=False) )


