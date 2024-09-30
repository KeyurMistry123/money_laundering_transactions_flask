from flask import Flask, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model, scaler, and label encoder from pickle files
print("Loading the model, scaler, and label encoder...")
with open('aml_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Load the dataset
print("Loading the transaction dataset...")
df = pd.read_csv('synthetic_aml_transactions.csv')

# Preprocess the dataset to match the model's input format
df['sender_name_encoded'] = label_encoder.transform(df['sender_name'])
df['receiver_name_encoded'] = label_encoder.transform(df['receiver_name'])
df['transaction_amount_scaled'] = scaler.transform(df[['transaction_amount']])

# Define the features used for prediction
features = ['sender_name_encoded', 'receiver_name_encoded', 'transaction_amount_scaled']

# Route for displaying transactions and predictions
@app.route('/transactions')
def transactions():
    # Generate model predictions
    print("Generating predictions...")
    df['is_fraudulent'] = model.predict(df[features])

    # Convert is_fraudulent to readable format
    df['prediction'] = df['is_fraudulent'].apply(lambda x: 'Fraudulent' if x == 1 else 'Non-Fraudulent')

    # Prepare data for frontend (HTML table)
    transactions_list = df[['sr_no', 'transaction_id', 'sender_name', 'receiver_name', 'transaction_amount', 'prediction']].to_dict(orient='records')

    return render_template('index.html', transactions=transactions_list)

# Root route
@app.route('/')
def home():
    return "Welcome to the Money Laundering Risk Score App!"

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
