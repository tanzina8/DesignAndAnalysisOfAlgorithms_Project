import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import time
import os


# Configuration
SOH_THRESHOLD = 0.6  # Threshold
CELL_COLUMNS = [f'U{i}' for i in range(1, 22)]
TARGET_COLUMN = 'SOH'




# Helper Functions


def plot_soh_comparison(y_test, y_pred, title):
   plt.figure(figsize=(8, 6))
   plt.scatter(y_test, y_pred, alpha=0.7, label="Predictions")


   # Add perfect prediction line (y = x)
   min_val = min(min(y_test), min(y_pred))
   max_val = max(max(y_test), max(y_pred))


   plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction (y=x)")


   # Add line of best fit
   m, b = np.polyfit(y_test, y_pred, 1)
   plt.plot(y_test, m * y_test + b, color='green', label="Regression Fit")


   # Detailed tick marks
   tick_start = max(0.60, np.floor(min_val * 10) / 10)
   tick_end = min(1.00, np.ceil(max_val * 10) / 10)
   xticks = np.arange(tick_start, tick_end + 0.01, 0.02)


   plt.xticks(xticks, rotation=45)
   plt.yticks(xticks)


   plt.xlabel("Actual SOH")
   plt.ylabel("Predicted SOH")
   plt.title(title)
   plt.legend()
   plt.grid(True, linestyle='--', alpha=0.6)
   plt.tight_layout()
   plt.show()




def train_and_evaluate(filepath: str, shuffle_split: bool, friendly_name: str):
   print(f"\n--- Training on {os.path.basename(filepath)} (Shuffle: {shuffle_split}) ---")


   try:
       data = pd.read_csv(filepath)
   except FileNotFoundError:
       print(f"Error: Dataset not found at {filepath}. Skipping this run.")
       return None, None, None


   # Define features (X) and target (y)
   X = data[CELL_COLUMNS]
   y = data[TARGET_COLUMN]


   # Split data (80% train, 20% test)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, shuffle=shuffle_split
   )


   # Train Linear Regression model
   model = LinearRegression()


   start_time = time.time()
   model.fit(X_train, y_train)
   training_time = time.time() - start_time


   # Predict on the test set
   y_pred = model.predict(X_test)


   # Evaluate the performance
   r2 = r2_score(y_test, y_pred)
   mse = mean_squared_error(y_test, y_pred)
   mae = mean_absolute_error(y_test, y_pred)


   # Apply 0.6 threshold
   predicted_classification = (y_pred >= SOH_THRESHOLD)
   actual_classification = (y_test >= SOH_THRESHOLD)


   classification_accuracy = np.mean(predicted_classification == actual_classification)


   metrics = {
       'display_name': friendly_name,
       'Split_Shuffled': shuffle_split,
       'R2_Score': r2,
       'MSE': mse,
       'MAE': mae,
       'Classification_Accuracy': classification_accuracy,
       'Training_Time_s': training_time
   }


   return metrics, y_test, y_pred




def get_health_status(soh_prediction: float):

   if soh_prediction < SOH_THRESHOLD:
       return "The battery has a problem."
   else:
       return "The battery is healthy."







if __name__ == "__main__":


   # List of training scenarios to compare
   scenarios = [
       # 1. Original shuffled split
       {'file': "data/PulseBat_processed_original.csv", 'shuffle': True,
        'display_name': '1. Original (Shuffled Split) - BEST APPROACH'},


       # 2. Descending data non shuffled split
       {'file': "data/PulseBat_processed_sorted_asc.csv", 'shuffle': False,
        'display_name': '2. SOH Ascending (Non-Shuffled Split)'},


       # 3. Descending data non shuffled split
       {'file': "data/PulseBat_processed_sorted_desc.csv", 'shuffle': False,
        'display_name': '3. SOH Descending (Non-Shuffled Split)'},
   ]


   results = []


   # Generate plots and run scenarios
   for scenario in scenarios:
       metrics, y_test, y_pred = train_and_evaluate(scenario['file'], scenario['shuffle'], scenario['display_name'])


       if metrics:
           results.append(metrics)
           plot_soh_comparison(y_test, y_pred, f"Predicted vs Actual SOH: {scenario['display_name']}")


   # Display comparison table
   if results:
       results_df = pd.DataFrame(results)
       results_df = results_df.round(4)


       results_df = results_df[
           ['display_name', 'Split_Shuffled', 'R2_Score', 'MSE', 'MAE', 'Classification_Accuracy', 'Training_Time_s']]


       print("\n" + "=" * 80)
       print("Model Performance Comparison (Impact of Data Sorting on Train/Test Split)")
       print("=" * 80)
       print(results_df.to_string(index=False))
       print("\n" + "=" * 80)



       best_result = results_df[results_df['display_name'].str.contains('BEST APPROACH')].iloc[0]
       print(f"\nModel Selected for Chatbot (Best R²: {best_result['R2_Score']:.4f}): Scenario 1")



       data_original = pd.read_csv("data/PulseBat_processed_original.csv")
       model = LinearRegression()
       X_train, X_test, y_train, y_test = train_test_split(
           data_original[CELL_COLUMNS], data_original[TARGET_COLUMN], test_size=0.2, random_state=42, shuffle=True
       )
       model.fit(X_train, y_train)



       sample_X = X_test.iloc[[0]]
       sample_y = y_test.iloc[0]


       predicted_soh = model.predict(sample_X)[0]
       status = get_health_status(predicted_soh)

       # print metrics
       print("\n" + "=" * 80)
       print("Example Battery Health Classification")
       print(f"Actual SOH: {sample_y:.4f}")
       print(f"Predicted SOH: {predicted_soh:.4f}")
       print(f"Classification Rule ({SOH_THRESHOLD}): {status}")
       print("=" * 80)

