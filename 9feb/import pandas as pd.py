import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

import os

# Prefer a local `day.csv` next to this script, fall back to working directory
local_csv = os.path.join(os.path.dirname(__file__), "day.csv")
if os.path.exists(local_csv):
    data_path = local_csv
elif os.path.exists("day.csv"):
    data_path = "day.csv"
else:
    # keep the original absolute path as a last resort for the original environment
    data_path = r"D:\GLA Mathura\Jan May 2026\Machine Learning Lab\Dimensionality reduction\day.csv"

try:
    data = pd.read_csv(data_path)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Could not find 'day.csv'. Tried: {data_path}") from e
print("Original Data Shape:", data.shape)
print(data.head())
missing_ratio = data.isnull().sum() / len(data) * 100
threshold = 20
selected_vars = [col for col in data.columns if missing_ratio[col] <= threshold]
data_mv = data[selected_vars]
print("\nAfter Missing Value Ratio:", data_mv.shape)

data_corr = data_mv.copy()
if 'ID' in data_corr.columns:
    data_corr = data_corr.drop(['ID'], axis=1)

data_corr = data_corr.select_dtypes(include=[np.number])
correlation = data_corr.corr()
high_corr = []
for c1 in correlation.columns:
    for c2 in correlation.columns:
        if c1 != c2 and c2 not in high_corr and correlation[c1][c2] > 0.9:
            high_corr.append(c1)

data_corr_filtered = data_corr.drop(high_corr, axis=1)
print("\nAfter High Correlation Filter:", data_corr_filtered.shape)

data_lv = data_corr_filtered.copy()
if 'ID' in data_lv.columns:
    data_lv = data_lv.drop('ID', axis=1)

data_lv = data_lv.select_dtypes(include=[np.number])
data_norm = normalize(data_lv)
data_scaled = pd.DataFrame(data_norm, columns=data_lv.columns)

variance = data_scaled.var()
threshold_var = 0.006
selected_vars = [col for col in data_lv.columns if variance[col] >= threshold_var]
data_lv_filtered = data_lv[selected_vars]
print("\nAfter Low Variance Filter:", data_lv_filtered.shape)

X = data_lv_filtered.select_dtypes(include=[np.number])
X = X.drop("cnt", axis=1, errors='ignore')
y = data_lv_filtered['cnt'] if 'cnt' in data_lv_filtered.columns else data['cnt']

model = RandomForestRegressor(random_state=1, max_depth=10)
model.fit(X, y)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8,5))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

selector = SelectFromModel(model, prefit=True)
X_selected = selector.transform(X)
print("\nFinal Shape after Random Forest Selection:", X_selected.shape)
