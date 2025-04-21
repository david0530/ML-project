# File no longer in use

import pandas as pd
from sklearn.model_selection import train_test_split

# Function to split the CSV file into training and testing sets
def split_csv(input_csv, train_csv, test_csv, test_size=0.2, random_state=42):
    # Load the input CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Save the training and testing data to separate CSV files
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Data has been split: {train_csv} (Training), {test_csv} (Testing)")

# Example usage:
input_csv = 'processed_segments_psd.csv'  # Path to your input CSV file
train_csv = 'train_data.csv'  # Path where the training CSV should be saved
test_csv = 'test_data.csv'    # Path where the testing CSV should be saved

split_csv(input_csv, train_csv, test_csv)
