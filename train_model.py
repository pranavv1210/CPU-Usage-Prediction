import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json
import numpy as np
import joblib

# Define the name of the dataset file
file_name = "data-1754297123597.csv"

# Load the dataset
df = pd.read_csv(file_name)

# Select the independent and dependent variables
independent_variables = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes']
dependent_variable = 'cpu_usage'

# Filter the DataFrame to only include the columns we need
df = df[independent_variables + [dependent_variable]].copy()

# Drop rows with missing values in the selected columns
df.dropna(subset=independent_variables, inplace=True)

# Define features (X) and target (y)
X = df.drop(columns=[dependent_variable])
y = df[dependent_variable]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor model
# n_estimators is the number of trees in the forest
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Store metrics in a dictionary
metrics = {
    "model": "RandomForestRegressor",
    "mse": mse,
    "rmse": rmse,
    "r2_score": r2
}

# Save the model to a file
model_output_path = "model.joblib"
joblib.dump(model, model_output_path)

# Save the metrics to a JSON file
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Model training and evaluation complete.")
print(f"Model saved to: {model_output_path}")
print("Metrics saved to: metrics.json")