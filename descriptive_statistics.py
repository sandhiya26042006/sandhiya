import pandas as pd   
df = pd.read_csv("delhi_air.csv")
numerical_cols=['Holidays_Count','Days','PM2.5','PM10','NO2','SO2','CO','Ozone','AQI']
print("Mean:\n",df[numerical_cols].mean())
print("Median:\n",df[numerical_cols].median())
print("Mode:\n",df[numerical_cols].mode())

print("Range:\n",df[numerical_cols].max()-df[numerical_cols].min())
print("Variance:\n",df[numerical_cols].var())
print("Standard Deviation:\n",df[numerical_cols].std())

print("Skewness:\n",df[numerical_cols].skew())
print("Kurtosis:\n",df[numerical_cols].kurt())

