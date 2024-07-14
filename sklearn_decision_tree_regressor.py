import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

# load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# splitting into features and the label
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# Recursively splits based on each feature, It doesn't use gradient descent
regressor = DecisionTreeRegressor()

# Use cross-validation to evaluate the model
scores = cross_val_score(regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mean_score = -scores.mean()
print(f"Cross-validated Mean Squared Error: {mean_score}")

# Hyperparameter tuning with GridSearchCV
param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# train the final model with the best parameters
best_regressor = grid_search.best_estimator_
best_regressor.fit(X_train, y_train)

# test the final model
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values
y_test_pred = best_regressor.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test Mean Squared Error: {test_mse}")
