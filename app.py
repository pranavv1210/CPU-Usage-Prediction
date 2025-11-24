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
            'runtime_minutes': [float(data['runtime_minutes'])]
        }
        
        df_processed = pd.DataFrame(input_data)
        
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
