import pandas as pd

# Load the PulseBat dataset from Excel
data = pd.read_excel("data/PulseBat Dataset.xlsx")
print("Dataset loaded successfully!")

# Preview the first 5 rows
print("\nFirst 5 rows:")
print(data.head())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Aggregate U1–U21 cells to calculate pack SOH
cell_columns = [f'U{i}' for i in range(1, 22)]
data['Pack_SOH'] = data[cell_columns].mean(axis=1)

# Save dataset for later training
data.to_csv("data/PulseBat_processed.csv", index=False)
print("\nProcessed dataset saved as 'PulseBat_processed.csv'")

