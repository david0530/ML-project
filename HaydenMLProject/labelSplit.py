import pandas as pd

# Function to separate labels and features from a CSV
def separate_labels(input_csv, labels_csv, features_csv):
    # Load the input CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Separate the labels (last column) and features (all other columns)
    labels = df.iloc[:, -1]  # The last column (Labels)
    features = df.iloc[:, :-1]  # All columns except the last one

    # Save the labels and features to separate CSV files
    labels.to_csv(labels_csv, index=False, header=True)  # Save labels as a CSV
    features.to_csv(features_csv, index=False, header=True)  # Save features as a CSV

    print(f"Labels saved to: {labels_csv}")
    print(f"Features saved to: {features_csv}")

# Example usage:
input_csv = 'training_data_flattened.csv'  # Path to your input CSV file
labels_csv = 'train_labels.csv'     # Path where the labels CSV should be saved
features_csv = 'train_features.csv' # Path where the features CSV should be saved

separate_labels(input_csv, labels_csv, features_csv)
