import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        # Convert types
        input_data = {
            'cpu_request': [float(data['cpu_request'])],
            'mem_request': [float(data['mem_request'])],
            'cpu_limit': [float(data['cpu_limit'])],
            'mem_limit': [float(data['mem_limit'])],
            'runtime_minutes': [float(data['runtime_minutes'])],
            'controller_kind': [data['controller_kind']]
        }
        
        df = pd.DataFrame(input_data)
        
        # One-hot encode controller_kind as done in training
        # We need to ensure all columns from training are present
        # The training script used: pd.get_dummies(df, columns=['controller_kind'], drop_first=True)
        # We need to match the columns expected by the model.
        
        # A robust way is to align with training columns. 
        # Since we don't have the training columns saved, we'll manually handle it based on known categories.
        # Known categories from dummy data gen: 'Deployment', 'StatefulSet', 'DaemonSet'
        # drop_first=True means one of them is dropped.
        # Let's check the model's feature names if possible, or just replicate the logic.
        
        # Replicating logic:
        # If we have a single row, get_dummies might not create all columns if the category isn't present.
        # So we should manually create the dummy columns.
        
        # Expected columns (based on get_dummies drop_first=True):
        # controller_kind_Deployment, controller_kind_StatefulSet (if DaemonSet is first alphabetically? No, get_dummies sorts)
        # Categories: DaemonSet, Deployment, StatefulSet.
        # drop_first=True drops DaemonSet.
        # So we expect: controller_kind_Deployment, controller_kind_StatefulSet.
        
        df_processed = pd.DataFrame()
        df_processed['cpu_request'] = df['cpu_request']
        df_processed['mem_request'] = df['mem_request']
        df_processed['cpu_limit'] = df['cpu_limit']
        df_processed['mem_limit'] = df['mem_limit']
        df_processed['runtime_minutes'] = df['runtime_minutes']
        
        # Add dummy columns
        df_processed['controller_kind_Deployment'] = (df['controller_kind'] == 'Deployment').astype(int)
        df_processed['controller_kind_StatefulSet'] = (df['controller_kind'] == 'StatefulSet').astype(int)
        # Note: If it's DaemonSet, both will be 0.
        
        # Ensure column order matches training (usually pandas preserves order, but let's be safe)
        # The training script:
        # independent_variables = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
        # get_dummies puts dummy cols at the end or replaces the original.
        # So order: cpu_request, mem_request, cpu_limit, mem_limit, runtime_minutes, controller_kind_Deployment, controller_kind_StatefulSet
        
        prediction = model.predict(df_processed)[0]
        
        return jsonify({'prediction': prediction})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
