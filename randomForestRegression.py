import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# READING THE DATASETS
filepath = r"C:\Users\Namratha\Documents\Informac"
os.chdir(filepath)

inspection_data = pd.read_csv(filepath + "/health.csv")

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
data = inspection_data.select_dtypes(include=numerics)
data = pd.DataFrame(data)
y = data['SCORE']
cols = ['SCORE', 'GRADE', 'Latitude', 'Longitude', 'FACILITY ZIP', 'Zip Codes', 'month_old', 'year', 'Total Violations']
data.drop(cols, axis = 1, inplace= True )
feature_list = list(data.columns)
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.3, random_state = 42)

#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)

#dt_clf = DecisionTreeRegressor(splitter="random", max_leaf_nodes=16, random_state=0)
#bag_clf = BaggingRegressor(dt_clf, n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=0)
#bag_clf.fit(X_train, y_train)
#y_pred = bag_clf.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

rnd_clf = RandomForestRegressor(n_estimators = 500)
rnd_clf.fit(X_train, y_train)

# Use the forest's predict method on the test data
predictions = rnd_clf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
#print('Mean absolute percentage error:', round(mape, 5), '%.')
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 4), '%.')

feature_imp = pd.Series(rnd_clf.feature_importances_, index = data.columns).sort_values(ascending=False)
#print(feature_imp)

sns.barplot(x=feature_imp.head(10), y=feature_imp.head(10).index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
