import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


# READING THE DATASETS
filepath = r"C:\Users\Namratha\Documents\Informac"
os.chdir(filepath)

inspection_data = pd.read_csv(filepath + "/health.csv")

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
data = inspection_data.select_dtypes(include=numerics)
data = pd.DataFrame(data)
y = data['GRADE']
cols = ['SCORE', 'GRADE', 'Latitude', 'Longitude', 'FACILITY ZIP', 'Zip Codes', 'month_old', 'year', 'Total Violations']
data.drop(cols, axis = 1, inplace= True )

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)

#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)

dt_clf = DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=0)
bag_clf = BaggingClassifier(dt_clf, n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=0)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

feature_imp = pd.Series(rnd_clf.feature_importances_, index = data.columns).sort_values(ascending=False)
#print(feature_imp)

sns.barplot(x=feature_imp.head(10), y=feature_imp.head(10).index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

#for name, score in zip(X_train, rnd_clf.feature_importances_):
    #print(name, score)
    #dict1 = {}
    #dict1[name] = score
    #rows_list.append(dict1)

