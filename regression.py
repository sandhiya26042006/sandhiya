
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("health.csv")

 
print("First 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = SimpleImputer(strategy='mean').fit_transform(df[num_cols])

# Encode categorical data
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

target_col = 'Outcome'   

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

df.hist(figsize=(8,6))
plt.show()

# Heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()

# ==============================
# 8. Decision Tree Regression
# ==============================
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ==============================
# 9. Evaluation
# ==============================
print("\n--- Decision Tree Regression ---")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print("\nFinal R2 Score:", r2_score(y_test, y_pred))