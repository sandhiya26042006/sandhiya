import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("delhi_air.csv")

# ========== 1. EXPLORATORY DATA ANALYSIS ==========

# Basic Info
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 Rows:\n", df.head())
print("\nStatistical Summary:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())
print("\nCorrelation Matrix:\n", df.corr(numeric_only=True))

# ========== 2. VISUALIZATION ==========

numerical_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'AQI']

# --- Histogram ---
df[numerical_cols].hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numerical Features", fontsize=14)
plt.tight_layout()
plt.show()

# --- Boxplot (to detect outliers) ---
plt.figure(figsize=(12, 6))
df[numerical_cols].boxplot()
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# --- Pairplot ---
sns.pairplot(df[numerical_cols[:4]])  # Limit to 4 cols for readability
plt.suptitle("Pairplot", y=1.02)
plt.show()

# --- Bar plot (if categorical column exists) ---
if 'Months' in df.columns:
    plt.figure(figsize=(10, 5))
    df.groupby('Month')['AQI'].mean().plot(kind='bar', color='teal', edgecolor='black')
    plt.title("Average AQI by Month")
    plt.xlabel("Month")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.show()

# --- Line plot ---
plt.figure(figsize=(12, 5))
plt.plot(df['AQI'], color='purple', linewidth=0.8)
plt.title("AQI Trend Over Time")
plt.xlabel("Index")
plt.ylabel("AQI")
plt.tight_layout()
plt.show()

