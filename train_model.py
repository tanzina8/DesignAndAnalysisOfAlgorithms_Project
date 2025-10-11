import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Load the processed dataset
data = pd.read_csv("data/PulseBat_processed.csv")
print("Processed dataset loaded successfully!")

# Feature columns and target
cell_cols = [f'U{i}' for i in range(1, 22)]
y = data['SOH']

data[cell_cols] = data[cell_cols].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=cell_cols + ['SOH'])

# Sorting methods
sorting_methods = {
    "original": lambda row: row.values,
    "ascending": lambda row: sorted(row.values),
    "descending": lambda row: sorted(row.values, reverse=True)
}

# Loop over sorting methods
for name, sort_func in sorting_methods.items():
    print(f"\n--- {name.upper()} SORTING ---")

    # Apply sorting to features
    X_sorted = data[cell_cols].apply(sort_func, axis=1, result_type='expand')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sorted, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete!")

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\nModel Evaluation Metrics (on test data):")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Threshold-based classification
    threshold = 0.6
    status = ["Healthy" if val >= threshold else "Problem" for val in y_pred]
    print("Sample classifications:", status[:10])

    # Plot Actual vs Predicted
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.7, label="Predictions")

    # Perfect prediction line
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Perfect Prediction")

    # Regression fit line (line of best fit)
    m, b = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test, m*y_test + b, color='green', label="Regression Fit")

    # Detailed ticks
    xticks = np.arange(0.60, 1.01, 0.02)
    yticks = np.arange(0.60, 1.01, 0.02)
    plt.xticks(xticks)
    plt.yticks(yticks)

    plt.xlabel("Actual SOH")
    plt.ylabel("Predicted SOH")
    plt.title(f"Predicted vs Actual Battery SOH ({name.capitalize()} sorting)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

