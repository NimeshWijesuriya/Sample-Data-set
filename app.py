from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from src.mlProject.pipeline.prediction import PredictionPipeline
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home_page():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def train():
    try:
        train_model()
        return "Training Successful!"
    except Exception as e:
        return f"Training failed: {e}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve and validate inputs
        gender = str(request.form.get('Customer Gender', ''))
        SeniorCitizen = int(request.form.get('Senior Citizen', 0))  # Convert to integer
        Partner = str(request.form.get('Partner', ''))
        Dependents = str(request.form.get('Dependents', ''))
        tenure = int(request.form.get('Network Stay', 0))  # Convert to integer
        PhoneService = str(request.form.get('Value Added Service', ''))
        MultipleLines = str(request.form.get('Multiple Lines', ''))
        InternetService = str(request.form.get('Internet Service', ''))
        OnlineSecurity = str(request.form.get('Online Security', ''))
        OnlineBackup = str(request.form.get('Online Backup', ''))
        DeviceProtection = str(request.form.get('Device Protection', ''))
        TechSupport = str(request.form.get('Tech Support', ''))
        StreamingTV = str(request.form.get('Streaming TV', ''))
        StreamingMovies = str(request.form.get('Streaming Movies', ''))
        Contract = str(request.form.get('Contract', ''))
        PaperlessBilling = str(request.form.get('Paperless Billing', ''))
        PaymentMethod = str(request.form.get('Payment Method', ''))
        MonthlyCharges = float(request.form.get('Monthly Charges', 0.0))  # Convert to float
        TotalCharges = str(request.form.get('Total Charges', '0.0'))  # Keep as string
        
        def categorize_tenure(tenure):
            if 1 <= tenure <= 12:
                return 'tenure_group_1 - 12'
            elif 13 <= tenure <= 24:
                return 'tenure_group_13 - 24'
            elif 25 <= tenure <= 36:
                return 'tenure_group_25 - 36'
            elif 37 <= tenure <= 48:
                return 'tenure_group_37 - 48'
            elif 49 <= tenure <= 60:
                return 'tenure_group_49 - 60'
            elif 61 <= tenure <= 72:
                return 'tenure_group_61 - 72'
            else:
                return 'tenure_group_61 - 72'  # Handle out-of-bounds or unknown values
        
        tenure_group = categorize_tenure(tenure)
        
        # Define categorical columns and possible values
        categorical_columns = {
            'Customer Gender': ['Female', 'Male'],
            'Partner': ['No', 'Yes'],
            'Dependents': ['No', 'Yes'],
            'Value Added Service': ['No', 'Yes'],
            'Multiple Lines': ['No', 'No phone service', 'Yes'],
            'Internet Service': ['DSL', 'Fiber optic', 'No'],
            'Online Security': ['No', 'No internet service', 'Yes'],
            'Online Backup': ['No', 'No internet service', 'Yes'],
            'Device Protection': ['No', 'No internet service', 'Yes'],
            'Tech Support': ['No', 'No internet service', 'Yes'],
            'Streaming TV': ['No', 'No internet service', 'Yes'],
            'Streaming Movies': ['No', 'No internet service', 'Yes'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Paperless Billing': ['No', 'Yes'],
            'Payment Method': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'],
            'tenure_group': ['tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36', 'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72']
        }
        
        # Create a DataFrame from the input data
        input_data = {
            'Customer Gender': [gender],
            'Senior Citizen': [SeniorCitizen],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'Network Stay': [tenure],
            'Value Added Service': [PhoneService],
            'Multiple Lines': [MultipleLines],
            'Internet Service': [InternetService],
            'Online Security': [OnlineSecurity],
            'Online Backup': [OnlineBackup],
            'Device Protection': [DeviceProtection],
            'Tech Support': [TechSupport],
            'Streaming TV': [StreamingTV],
            'Streaming Movies': [StreamingMovies],
            'Contract': [Contract],
            'Paperless Billing': [PaperlessBilling],
            'Payment Method': [PaymentMethod],
            'Monthly Charges': [MonthlyCharges],
            'Total Charges': [TotalCharges],
            'tenure_group': [tenure_group]  # Include tenure group
        }
        df = pd.DataFrame(input_data)
        
        # Convert categorical variables to dummy variables
        df_encoded = pd.get_dummies(df, columns=categorical_columns.keys())
        
        # Ensure all possible dummy variables are present
        all_columns = []
        for cat_col, categories in categorical_columns.items():
            all_columns.extend([f"{cat_col}_{cat}" for cat in categories])
        
        all_columns.extend(['Senior Citizen', 'Network Stay', 'Monthly Charges', 'Total Charges'])
        
        # Add missing columns with zero
        for col in all_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[all_columns]
        
        # Drop the 'Network Stay' column if it is included in the encoded columns
        if 'Network Stay' in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=['Network Stay'])
        
        # Normalize the features
        scaler = MinMaxScaler()
        df_encoded = scaler.fit_transform(df_encoded)
        
        # Convert to numpy array
        data = np.array(df_encoded)
        
        # Predict
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(data)

        return render_template('results.html', prediction=str(prediction))

    except Exception as e:
        import traceback
        error_message = f"Prediction error: {e}\n{traceback.format_exc()}"
        print(error_message)
        return f"An error occurred during prediction: {error_message}"

def train_model():
    # Placeholder for actual training code
    pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
