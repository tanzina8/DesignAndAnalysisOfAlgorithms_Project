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

# Aggregate U1–U21 cells to calculate battery pack SOH
cell_columns = [f'U{i}' for i in range(1, 22)]
data['Pack_SOH'] = data[cell_columns].mean(axis=1)

# Save Original Processed
data.to_csv("data/PulseBat_processed_original.csv", index=False)
print("Processed dataset saved as 'PulseBat_processed_original.csv'")

# Save Ascending Sorted
data_sorted_asc = data.sort_values(by='Pack_SOH', ascending=True)
data_sorted_asc.to_csv("data/PulseBat_processed_sorted_asc.csv", index=False)
print("Ascending sorted dataset saved as 'PulseBat_processed_sorted_asc.csv'")

# Save Descending Sorted
data_sorted_desc = data.sort_values(by='Pack_SOH', ascending=False)
data_sorted_desc.to_csv("data/PulseBat_processed_sorted_desc.csv", index=False)
print("Descending sorted dataset saved as 'PulseBat_processed_sorted_desc.csv'")

