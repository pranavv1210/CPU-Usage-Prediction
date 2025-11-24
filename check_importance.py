import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('model.joblib')

# Get feature importances
importances = model.feature_importances_

# Get feature names
# We need to reconstruct the feature names as they were during training
# The order was: cpu_request, mem_request, cpu_limit, mem_limit, runtime_minutes, controller_kind_Deployment, controller_kind_StatefulSet
feature_names = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind_Deployment', 'controller_kind_StatefulSet']

# Create a DataFrame
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print("Feature Importances:")
print(feature_importance_df)
