import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv("train_processed_segments_psd.csv").drop("TD", axis=1)
test_df = pd.read_csv("test_processed_segments_psd.csv").drop("TD", axis=1)

x_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"]

x_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

train_corr = train_df.corr(method = "pearson")

plt.figure(figsize = (9,9))
sns.heatmap(train_corr, xticklabels=True, yticklabels=True)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.title("Train data pearson correlation heatmap")
plt.savefig("Heatmap.png", dpi=300)

test_corr = train_df.corr(method = "pearson")

plt.figure(figsize = (9,9))
sns.heatmap(test_corr, xticklabels=True, yticklabels=True)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.title("Test data pearson correlation heatmap")
plt.show()
