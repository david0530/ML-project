import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import numpy as np

# Load the data
features = pd.read_csv('train_features.csv')
features = features.drop(features.columns[[0, 1]], axis=1)
labels = pd.read_csv('train_labels.csv').values.ravel()  # Flatten to 1D

# Set initial parameters for XGBoost
base_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.1,
    'seed': 42
}


# Hyperparameter grids
max_depth_values = [3, 6, 10]
learning_rate_values = [0.01, 0.1, 0.2]
subsample_values = [0.7, 0.8, 1]
colsample_bytree_values = [0.7, 0.8, 1]

def evaluate_hyperparameter(param_name, param_values, base_params, plot_title):
    train_rmse_values, val_rmse_values = [], []

    for value in param_values:
        params = base_params.copy()
        params[param_name] = value

        # Lists to hold metrics for each fold
        fold_train_rmse, fold_val_rmse = [], []
        fold_train_mae, fold_val_mae = [], []
        fold_train_r2, fold_val_r2 = [], []

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, val_index in kf.split(features):
            X_train, X_val = features.iloc[train_index], features.iloc[val_index]
            y_train, y_val = labels[train_index], labels[val_index]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            bst = xgb.train(params, dtrain, num_boost_round=100,
                            evals=[(dval, 'eval')],
                            early_stopping_rounds=10,
                            verbose_eval=False)

            y_train_pred = bst.predict(dtrain)
            y_val_pred = bst.predict(dval)

            # Compute and store metrics
            fold_train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
            fold_val_rmse.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))

            fold_train_mae.append(mean_absolute_error(y_train, y_train_pred))
            fold_val_mae.append(mean_absolute_error(y_val, y_val_pred))

            fold_train_r2.append(r2_score(y_train, y_train_pred))
            fold_val_r2.append(r2_score(y_val, y_val_pred))

        # Means and STDs
        train_rmse_mean = np.mean(fold_train_rmse)
        train_rmse_std = np.std(fold_train_rmse)

        val_rmse_mean = np.mean(fold_val_rmse)
        val_rmse_std = np.std(fold_val_rmse)

        train_mae_mean = np.mean(fold_train_mae)
        train_mae_std = np.std(fold_train_mae)

        val_mae_mean = np.mean(fold_val_mae)
        val_mae_std = np.std(fold_val_mae)

        train_r2_mean = np.mean(fold_train_r2)
        train_r2_std = np.std(fold_train_r2)

        val_r2_mean = np.mean(fold_val_r2)
        val_r2_std = np.std(fold_val_r2)

        train_rmse_values.append(train_rmse_mean)
        val_rmse_values.append(val_rmse_mean)

        # Print metrics
        print(f"{param_name} = {value}")
        print(f"  Train -> RMSE: {train_rmse_mean:.4f} ± {train_rmse_std:.4f}, "
              f"MAE: {train_mae_mean:.4f} ± {train_mae_std:.4f}, "
              f"R²: {train_r2_mean:.4f} ± {train_r2_std:.4f}")
        print(f"  Val   -> RMSE: {val_rmse_mean:.4f} ± {val_rmse_std:.4f}, "
              f"MAE: {val_mae_mean:.4f} ± {val_mae_std:.4f}, "
              f"R²: {val_r2_mean:.4f} ± {val_r2_std:.4f}")
        print("")

    # Plotting RMSE means
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, train_rmse_values, label='Training RMSE', marker='o')
    plt.plot(param_values, val_rmse_values, label='Validation RMSE', marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Run evaluation for each hyperparameter
evaluate_hyperparameter('max_depth', max_depth_values, base_params, 'RMSE vs Max Depth (5-Fold CV)')
evaluate_hyperparameter('learning_rate', learning_rate_values, base_params, 'RMSE vs Learning Rate (5-Fold CV)')
evaluate_hyperparameter('subsample', subsample_values, base_params, 'RMSE vs Subsample (5-Fold CV)')
evaluate_hyperparameter('colsample_bytree', colsample_bytree_values, base_params, 'RMSE vs Colsample by Tree (5-Fold CV)')
