import pandas as pd
import numpy as np
from scipy import stats

url = "https://archive.ics.uci.edu/static/public/352/data.csv"
df = pd.read_csv(url)
print(df.head())

# Drop rows with missing values
df.dropna(inplace=True)
numeric_columns = ['Quantity', 'UnitPrice', 'CustomerID']
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Handling duplicates
df.drop_duplicates(inplace=True)
df
