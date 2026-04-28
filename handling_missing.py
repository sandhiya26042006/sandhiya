import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("delhi_air.csv")

  
print("Missing Values:\n", df.isnull().sum())
print("\nTotal Missing:", df.isnull().sum().sum())


imputer = SimpleImputer(strategy='mean')
df[['PM2.5', 'PM10']] = imputer.fit_transform(df[['PM2.5', 'PM10']])

print("\nAfter Imputation:\n", df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("After Removing Duplicates:", df.shape)


numerical_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'AQI']

std_scaler = StandardScaler()
df_standardized = pd.DataFrame(
    std_scaler.fit_transform(df[numerical_cols]),
    columns=numerical_cols
)
print("\nAfter StandardScaler (first 5 rows):\n", df_standardized.head())


mm_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    mm_scaler.fit_transform(df[numerical_cols]),
    columns=numerical_cols
)
print("\nAfter MinMaxScaler (first 5 rows):\n", df_normalized.head())


