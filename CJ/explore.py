import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv("train_processed_segments_psd.csv").drop("TD", axis=1)
test_df = pd.read_csv("test_processed_segments_psd.csv").drop("TD", axis=1)

x_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"]

x_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

plt.figure(figsize = (9,9))
sns.histplot(y_train, bins = 30)
plt.title("Histogram of labels")
plt.savefig("hist.png", dpi=300)


