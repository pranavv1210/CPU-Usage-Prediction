import pandas as pd
import numpy as np

# Define the number of samples
n_samples = 100

# Generate dummy data
data = {
    'cpu_request': np.random.randint(100, 1000, n_samples),
    'mem_request': np.random.randint(100, 1000, n_samples),
    'cpu_limit': np.random.randint(1000, 2000, n_samples),
    'mem_limit': np.random.randint(1000, 2000, n_samples),
    'runtime_minutes': np.random.randint(1, 100, n_samples),
    'controller_kind': np.random.choice(['Deployment', 'StatefulSet', 'DaemonSet'], n_samples),
    'cpu_usage': np.random.uniform(0, 1, n_samples)  # Assuming normalized usage or similar
}

df = pd.DataFrame(data)

# Save to CSV
file_name = "data-1754297123597.csv"
df.to_csv(file_name, index=False)

print(f"Dummy data generated and saved to {file_name}")
