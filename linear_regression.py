import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========== 1. LOAD DATASET ==========
df = pd.read_csv("delhi_air.csv")

print("Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())

# ========== 2. FEATURE & TARGET SELECTION ==========
# Goal: Predict AQI (continuous value) using other features
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']]
y = df['AQI']  # Target - continuous variable

print("\nTarget (AQI) Statistics:\n", y.describe())

# ========== 3. TRAIN-TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# ========== 4. MODEL - LINEAR REGRESSION ==========
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ========== 5. EVALUATION ==========
print("\n===== REGRESSION METRICS =====")
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))

# Coefficients
print("\nFeature Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# ========== 6. VISUALIZATION ==========

# Actual vs Predicted scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.tight_layout()
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.6, color='purple', edgecolors='black')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted AQI")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()

# ========== INFERENCE ==========
"""
INFERENCE:
1. Linear Regression predicts a CONTINUOUS value (AQI), unlike classification which predicts categories.
2. R² Score indicates how well the model explains variance (closer to 1 = better).
3. MAE tells average prediction error in same units as target (AQI).
4. RMSE penalizes large errors more than MAE.
5. The Actual vs Predicted plot shows points close to the red diagonal = good predictions.
6. The Residual plot should show random scatter around 0 (no pattern = good model).
7. Feature coefficients show which pollutants have the most impact on AQI.
8. Key difference: Regression predicts continuous values, Classification predicts categories.
"""
