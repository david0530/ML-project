import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FormatStrFormatter

df = pd.read_csv("combined_data_flattened.csv").iloc[:, 2:]
X = df.drop("Bayley raw gross motor", axis=1)
y = df["Bayley raw gross motor"]

from sklearn.svm import SVR

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

c_values = np.logspace(-1, 4, 50)
mse_c = []

for c in c_values:
    model = SVR(kernel='rbf', C=c, gamma='scale', epsilon=0.1)
    model.fit(X_train, y_train)
    mse_c.append(mean_squared_error(y_test, model.predict(X_test)))

plt.figure(figsize=(9, 9))
sns.lineplot(x=c_values, y=mse_c)
plt.xscale('log')
plt.title('SVR C Parameter Tuning (gamma="scale")')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Test MSE')
plt.grid(True)
plt.show()

gamma_values = np.logspace(-3, 2, 50)
mse_gamma = []

for gamma in gamma_values:
    model = SVR(kernel='rbf', C=1.0, gamma=gamma, epsilon=0.1)
    model.fit(X_train, y_train)
    mse_gamma.append(mean_squared_error(y_test, model.predict(X_test)))

plt.figure(figsize=(9, 9))
sns.lineplot(x=gamma_values, y=mse_gamma)
plt.xscale('log')
plt.title('SVR Gamma Parameter Tuning (C=1.0)')
plt.xlabel('Kernel Coefficient (gamma)')
plt.ylabel('Test MSE')
plt.grid(True)
plt.show()

# param_grid = {
#     'C': np.logspace(-1, 4, 50),
#     'gamma': np.logspace(-3, 2, 50)
# }
#
# results = []
# for c in param_grid['C']:
#     for gamma in param_grid['gamma']:
#         model = SVR(kernel='rbf', C=c, gamma=gamma, epsilon=0.1)
#         model.fit(X_train, y_train)
#         mse = mean_squared_error(y_test, model.predict(X_test))
#         results.append({'C': c, 'gamma': gamma, 'MSE': mse})
#
# results_df = pd.DataFrame(results)
# heatmap_data = results_df.pivot(index='C', columns='gamma', values='MSE')
#
# plt.figure(figsize=(9, 9))
# sns.heatmap(heatmap_data, cmap='viridis', annot=False, 
#             cbar_kws={'label': 'MSE'}, norm=LogNorm())
# plt.title('SVR C-Gamma Interaction Heatmap')
# plt.xlabel('Gamma (log)')
# plt.ylabel('C (log)')
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# plt.savefig("bigboy.png", dpi=300)

model = SVR(kernel='rbf', C=25, gamma=25, epsilon=0.1)
model.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
rmse_scorer = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False)

r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
mae_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring=mae_scorer)
rmse_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring=rmse_scorer)

print(f"R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
print(f"MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
print(f"RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
