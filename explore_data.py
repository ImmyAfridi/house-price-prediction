
from pathlib import Path
import pandas as pd
import kagglehub

# Locate the downloaded dataset
dataset_path = Path(
    kagglehub.dataset_download("yasserh/housing-prices-dataset")
)
csv_path = dataset_path / "Housing.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Explore the dataset
print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset dimensions (rows, columns) ---")
print(df.shape)

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- Data types and missing values ---")
df.info()

print("\n--- Missing values per column ---")
print(df.isnull().sum())

print("\n--- Numerical summary ---")
print(df.describe())