import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Load the processed dataset
data = pd.read_csv("data/PulseBat_processed.csv")
print("Processed dataset loaded successfully!")

# Include (U1–U21) and (SOH)
X = data[[f'U{i}' for i in range(1, 22)]]
y = data['SOH']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete!")

# Predict on the test set of data
y_pred = model.predict(X_test)

# Evaluate the performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Evaluation Metrics (on test data):")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Plot Actual vs Predicted SOH (edited ticks)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, label="Predictions")

# Add perfect prediction line (y = x)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Perfect Prediction")

# Add regression fit line (line of best fit)
m, b = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, m*y_test + b, color='green', label="Regression Fit")

# Detailed tick marks
xticks = np.arange(0.60, 1.01, 0.02)
yticks = np.arange(0.60, 1.01, 0.02)
plt.xticks(xticks)
plt.yticks(yticks)

plt.xlabel("Actual SOH")
plt.ylabel("Predicted SOH")
plt.title("Predicted vs Actual Battery SOH (Detailed Axis Ticks)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
