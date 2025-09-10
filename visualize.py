import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np # This line was missing

# Define file names
data_file = "data-1754297123597.csv"
model_file = "model.joblib"

# Load the trained model and the dataset
model = joblib.load(model_file)
df = pd.read_csv(data_file)

# Prepare the data just like in train_model.py
independent_variables = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
dependent_variable = 'cpu_usage'

# Filter and clean the data
df = df[independent_variables + [dependent_variable]].copy()
df.dropna(subset=independent_variables, inplace=True)
df = pd.get_dummies(df, columns=['controller_kind'], drop_first=True)

X = df.drop(columns=[dependent_variable])
y = df[dependent_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions using the loaded model
y_pred = model.predict(X_test)

# Create the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.5, s=5)
plt.title('Predicted vs. Actual CPU Usage', fontsize=16)
plt.xlabel('Actual CPU Usage', fontsize=12)
plt.ylabel('Predicted CPU Usage', fontsize=12)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_file = "predicted_vs_actual.png"
plt.savefig(plot_file)
print(f"Plot saved to: {plot_file}")

# Calculate residuals
residuals = y_test - y_pred

# Create the residual plot
plt.figure(figsize=(10, 8))
plt.scatter(y_pred, residuals, alpha=0.5, s=5)
plt.title('Residual Plot', fontsize=16)
plt.xlabel('Predicted CPU Usage', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Residuals')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("residual_plot.png")
print("Residual plot saved to: residual_plot.png")

# Get feature importances from the trained model
feature_importances = model.feature_importances_

# Get the feature names
feature_names = X.columns

# Sort features by importance
sorted_indices = np.argsort(feature_importances)[::-1]

# Create the bar chart
plt.figure(figsize=(12, 8))
plt.title('Feature Importance', fontsize=16)
plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature importance plot saved to: feature_importance.png")

# Create a histogram of the target variable
plt.figure(figsize=(10, 8))
plt.hist(y, bins=50, edgecolor='black', alpha=0.7)
plt.title('Distribution of CPU Usage', fontsize=16)
plt.xlabel('CPU Usage', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig("cpu_usage_distribution.png")
print("CPU usage distribution plot saved to: cpu_usage_distribution.png")