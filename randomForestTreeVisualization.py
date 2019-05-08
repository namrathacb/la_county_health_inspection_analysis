import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot

# READING THE DATASETS
filepath = r"C:\Users\Namratha\Documents\Informac"
os.chdir(filepath)

inspection_data = pd.read_csv(filepath + "/health.csv")

cols = ['SCORE','VIOLATION CODE_F007', 'VIOLATION CODE_F050', 'VIOLATION CODE_F054', 'VIOLATION CODE_F014', 'VIOLATION CODE_F023',
        'VIOLATION CODE_F006', 'VIOLATION CODE_F035', 'VIOLATION CODE_F052', 'VIOLATION CODE_F036', 'VIOLATION CODE_F033']
data = inspection_data[cols]
data = pd.DataFrame(data)
y = data['SCORE']
data.drop('SCORE', axis = 1, inplace=True)

feature_list = list(data.columns)
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.3, random_state = 42)

rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(X_train, y_train)

# Extract the small tree
tree_small = rf_small.estimators_[5]

# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');