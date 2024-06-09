import pandas

X = data[some_features <- type list] #gives subset of data (some columns)

-------------------
import sklearn.tree
model = DecisionTreeRegressor(max_leaf_nodes = None, random_state = 1) #defines a model with (none = unlimited) number of leaves
model.fit(X,y) # captures patterns ??
model.predict(X.head()) #predicts y feature based on features of X

import sklearn.metrics
mean_absolute_error(a,b) # average(|a_i-b_i|)

import sklearn.model_selection
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 0) #split data into training and validation data

import sklearn.ensemble 
forest_model = RandomForestRegressor(random_state = 1) # averages the prediction from each component tree

