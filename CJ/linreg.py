from statistics import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("combined_data_flattened.csv").iloc[:, 2:]
X = df.drop("Bayley raw gross motor", axis=1)
y = df["Bayley raw gross motor"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = np.logspace(-2, 4, 100)

mse_values = []
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse_values.append(mean_squared_error(y_test, y_pred))

results_df = pd.DataFrame({
    'Alpha': alphas,
    'MSE': mse_values
})

plt.figure(figsize=(9, 9))
sns.lineplot(data=results_df, x='Alpha', y='MSE')
plt.xscale('log')
plt.title('Ridge Regression Alpha Tuning')
plt.xlabel('Regularization Strength (Alpha) (log)')
plt.ylabel('Test MSE')
plt.grid(True)
# plt.savefig("pic.png", dpi=300)

model = Ridge(alpha=0.2)
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

