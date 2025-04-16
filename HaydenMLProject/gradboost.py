import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error  # Import mean_squared_error
from sklearn.model_selection import train_test_split  # Import train_test_split
import numpy as np  # For calculating RMSE

# Load the data
train_features = pd.read_csv('train_features.csv')
train_features = train_features.drop(train_features.columns[[0,1]], axis=1)
train_labels = pd.read_csv('train_labels.csv')

test_features = pd.read_csv('test_features.csv')
test_features = test_features.drop(test_features.columns[[0,1]], axis=1)
test_features = test_features.dropna()
test_labels = pd.read_csv('test_labels.csv')

# Convert the labels to numpy arrays (XGBoost expects a 1D array for labels)
train_labels = train_labels.values.ravel()
test_labels = test_labels.values.ravel()

# Split train data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(test_features, label=test_labels)

# Set initial parameters for XGBoost
params = {
    'objective': 'reg:squarederror',  # Regression problem
    'eval_metric': 'rmse',            # Root Mean Squared Error as evaluation metric
    'learning_rate': 0.1,             # Step size at each iteration
    'n_estimators': 100,              # Number of trees
    'seed': 42
}

# Define hyperparameter values to try for max_depth, learning_rate, subsample, and colsample_bytree
max_depth_values = [3, 6, 10]
learning_rate_values = [0.01, 0.1, 0.2]
subsample_values = [0.7, 0.8, 1]
colsample_bytree_values = [0.7, 0.8, 1]

# Store results
def evaluate_hyperparameter(param_name, param_values, params, plot_title):
    """
    A helper function to loop over hyperparameter values, train the model, and plot the RMSE scores.
    """
    train_rmse_values = []
    test_rmse_values = []
    
    for value in param_values:
        # Update the hyperparameter in params
        params[param_name] = value
        
        # Train the XGBoost model with the current hyperparameter
        bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'eval')], early_stopping_rounds=10)
        
        # Predict on the training set and the test set
        y_train_pred = bst.predict(dtrain)
        y_test_pred = bst.predict(dtest)
        
        # Calculate Root Mean Squared Error (RMSE)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(test_labels, y_test_pred))
        
        # Append results
        train_rmse_values.append(train_rmse)
        test_rmse_values.append(test_rmse)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, train_rmse_values, label='Training RMSE', marker='o')
    plt.plot(param_values, test_rmse_values, label='Test RMSE', marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Test and plot for each hyperparameter

# 1. Max Depth
evaluate_hyperparameter('max_depth', max_depth_values, params, 'RMSE vs Max Depth for XGBoost Model')

# 2. Learning Rate
evaluate_hyperparameter('learning_rate', learning_rate_values, params, 'RMSE vs Learning Rate for XGBoost Model')

# 3. Subsample
evaluate_hyperparameter('subsample', subsample_values, params, 'RMSE vs Subsample for XGBoost Model')

# 4. Colsample by Tree
evaluate_hyperparameter('colsample_bytree', colsample_bytree_values, params, 'RMSE vs Colsample by Tree for XGBoost Model')
