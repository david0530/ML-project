import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
train_features = pd.read_csv('train_features.csv')
train_features = train_features.drop(train_features.columns[0], axis=1)
train_labels = pd.read_csv('train_labels.csv')

test_features = pd.read_csv('test_features.csv')
test_features = test_features.drop(test_features.columns[0], axis=1)
test_features = test_features.dropna()
test_labels = pd.read_csv('test_labels.csv')

# Convert the labels to numpy arrays (XGBoost expects a 1D array for labels)
train_labels = train_labels.values.ravel()
test_labels = test_labels.values.ravel()

# Split train data into train and validation sets (optional for cross-validation)
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Create DMatrix for XGBoost (optional for more efficient training)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(test_features, label=test_labels)

# Set parameters for XGBoost
params = {
    'objective': 'reg:squarederror',  # Regression problem
    'eval_metric': 'rmse',            # Root Mean Squared Error as evaluation metric
    'max_depth': 6,                   # Maximum depth of the tree
    'learning_rate': 0.1,             # Step size at each iteration
    'n_estimators': 100,              # Number of trees
    'subsample': 0.8,                 # Fraction of samples to use for fitting trees
    'colsample_bytree': 0.8,          # Fraction of features to consider when building each tree
    'seed': 42
}

# Train the XGBoost model
bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'eval')], early_stopping_rounds=10)

# Predict on the training set and the test set
y_train_pred = bst.predict(dtrain)
y_test_pred = bst.predict(dtest)

# Evaluate the model using RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(test_labels, y_test_pred))

# Calculate R² score (a measure of accuracy for regression)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(test_labels, y_test_pred)

# Print evaluation results
print(f'Training RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')
print(f'Training R²: {train_r2:.2f}')
print(f'Test R²: {test_r2:.2f}')

# If you want to tune the hyperparameters using GridSearchCV, you can do so as follows:
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
# param_grid = {
#     'max_depth': [3, 6, 10],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [100, 200],
#     'subsample': [0.7, 0.8, 1],
#     'colsample_bytree': [0.7, 0.8, 1]
# }
# grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)
# print("Best hyperparameters:", grid_search.best_params_)
