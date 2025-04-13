import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("train_processed_segments_psd.csv").drop("TD", axis=1)
test_df = pd.read_csv("test_processed_segments_psd.csv").drop("TD", axis=1)

x_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"]
y_train = (y_train > 30).astype(int) #30 value found as reasonable splitting point

x_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]
y_test = (y_test > 30).astype(int) #30 value found as reasonable splitting point

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

### REG STRENGTH ###
reg_strength = [0.001, 0.01, 0.1, 1, 10, 100]
accuracy = []

for c in reg_strength:
    log_reg = LogisticRegression(C = c)
    log_reg.fit(x_train_scaled, y_train)

    y_pred = log_reg.predict(x_test_scaled)
    accuracy.append(accuracy_score(y_test, y_pred))

results_df = pd.DataFrame({
    'Regularization Strength (C)': reg_strength,
    'Accuracy': accuracy
})

# Plot
plt.figure(figsize=(9, 9))
sns.lineplot(x='Regularization Strength (C)', y='Accuracy', data=results_df, marker='o')
plt.xscale('log')
plt.title('Effect of Regularization Strength for Logistic Regression')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.grid()
plt.show()

### ROC Curve ###
lr = LogisticRegression(C = 0.01)
lr.fit(x_train_scaled, y_train)

y_pred_prob = lr.predict_proba(x_test_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(9, 9))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc='lower right')
plt.grid()
plt.show()

