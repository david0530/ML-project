import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import math

# --- Configuration ---
DATA_FILE_PATH = '/projects/dyang97/DGCNN/GCN/processed_segments_psd.csv'
LABEL_COLUMN = 'Label'
GROUP_COLUMNS = ['TD', 'Month'] # Columns defining the groups
# Columns to drop (identifiers/metadata not used as features + Label)
# Note: We keep GROUP_COLUMNS initially to create the groups variable
COLUMNS_TO_DROP_FOR_FEATURES = ['TD', 'Month', 'Segment', LABEL_COLUMN]
RANDOM_STATE = 42 # Seed for XGBoost, not used for GroupKFold splitting itself
NUM_BOOST_ROUND = 1000 # Fixed number of boosting rounds (trees) per fold
N_SPLITS = 5 # Number of folds for GroupKFold

# --- 1. Load the data ---
print(f"Loading data from: {DATA_FILE_PATH}")
try:
    df = pd.read_csv(DATA_FILE_PATH)
    print("Data loaded successfully.")
    print(f"Original dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Data Preprocessing ---
print("Preprocessing data...")
# Handle potential missing values
initial_rows = df.shape[0]
df.dropna(inplace=True)
if df.shape[0] < initial_rows:
    print(f"Dropped {initial_rows - df.shape[0]} rows containing NaN values.")
    print(f"Dataset shape after dropping NaNs: {df.shape}")

# Reset index just in case dropna messed it up, important for iloc later
df.reset_index(drop=True, inplace=True)

# --- 3. Define Groups and Separate Features/Labels ---
print(f"Defining groups based on columns: {GROUP_COLUMNS}")
# Create a unique group identifier for each combination of TD and Month
groups = df[GROUP_COLUMNS[0]].astype(str) + '_' + df[GROUP_COLUMNS[1]].astype(str)
n_unique_groups = groups.nunique()
print(f"Found {n_unique_groups} unique groups.")

# Separate features (X) and the target label (y)
if LABEL_COLUMN not in df.columns:
    print(f"Error: Label column '{LABEL_COLUMN}' not found in the dataset.")
    exit()

y = df[LABEL_COLUMN].values # Use .values for numpy array immediately

# Select feature columns (all columns except the ones specified)
feature_columns = [col for col in df.columns if col not in COLUMNS_TO_DROP_FOR_FEATURES]
X = df[feature_columns] # X is now a DataFrame

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Groups shape: {groups.shape}") # Should match y shape
print(f"Using {len(feature_columns)} features.")

# --- 4. Set up GroupKFold Cross-Validation ---
print(f"\nSetting up GroupKFold with {N_SPLITS} splits...")
gkf = GroupKFold(n_splits=N_SPLITS)

# Lists to store results from each fold
test_rmses = []
test_r2s = []
train_rmses = [] 
train_r2s = []

# --- 5. Iterate through Folds, Train, and Evaluate ---
print("Starting cross-validation...")

for fold, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    # Get data for this fold using the indices provided by GroupKFold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    groups_train, groups_test = groups.iloc[train_index], groups.iloc[test_index] # Keep track of groups

    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print(f"Number of groups in Train: {groups_train.nunique()}, Test: {groups_test.nunique()}")

    # --- Optional: Sanity check group separation ---
    train_groups_set = set(groups_train)
    test_groups_set = set(groups_test)
    assert len(train_groups_set.intersection(test_groups_set)) == 0, \
        f"Error: Groups found in both train and test sets in fold {fold+1}!"
    # print("Group separation check passed.") # Uncomment for confirmation

    # Create DMatrix for this fold
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    print("DMatrix created for this fold.")

    # Define XGBoost parameters (can be defined once outside the loop too)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': RANDOM_STATE # Use same seed for reproducibility within XGBoost runs
    }

    # Train the model for fixed rounds
    print(f"Training model for {NUM_BOOST_ROUND} rounds...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        # Evaluate on test split of *this fold* during training (optional)
        evals=[(dtrain, 'train'), (dtest, 'test')],
        verbose_eval= 200 # Print progress less frequently inside the loop
    )
    print("Training complete for this fold.")

    # Predict on train and test sets for this fold
    y_train_pred = bst.predict(dtrain)
    y_test_pred = bst.predict(dtest)

    # Evaluate performance for this fold
    fold_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    fold_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    fold_train_r2 = r2_score(y_train, y_train_pred)
    fold_test_r2 = r2_score(y_test, y_test_pred)

    # Store results
    train_rmses.append(fold_train_rmse)
    test_rmses.append(fold_test_rmse)
    train_r2s.append(fold_train_r2)
    test_r2s.append(fold_test_r2)

    # Print fold results
    print(f"Fold {fold+1} Train RMSE: {fold_train_rmse:.4f}, Test RMSE: {fold_test_rmse:.4f}")
    print(f"Fold {fold+1} Train R2:   {fold_train_r2:.4f}, Test R2:   {fold_test_r2:.4f}")

# --- 6. Aggregate and Print Final Results ---
print("\n--- Cross-Validation Summary ---")

avg_train_rmse = np.mean(train_rmses)
std_train_rmse = np.std(train_rmses)
avg_test_rmse = np.mean(test_rmses)
std_test_rmse = np.std(test_rmses)

avg_train_r2 = np.mean(train_r2s)
std_train_r2 = np.std(train_r2s)
avg_test_r2 = np.mean(test_r2s)
std_test_r2 = np.std(test_r2s)

print(f"Average Train RMSE: {avg_train_rmse:.4f} +/- {std_train_rmse:.4f}")
print(f"Average Test RMSE:  {avg_test_rmse:.4f} +/- {std_test_rmse:.4f}")
print(f"Average Train R2:   {avg_train_r2:.4f} +/- {std_train_r2:.4f}")
print(f"Average Test R2:    {avg_test_r2:.4f} +/- {std_test_r2:.4f}")
print("-------------------------------")

print("\nScript finished.")