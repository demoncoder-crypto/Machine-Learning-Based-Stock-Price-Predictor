from flask import Flask, render_template, request, jsonify
from model_utils import *
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        model_type = request.form['model_type']
        
        # Get and prepare data
        data = get_stock_data(ticker)
        X, y, scaler = preprocess_data(data)
        
        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train and predict
        if model_type == 'lstm':
            model = train_lstm_model(X_train, y_train)
            predictions = model.predict(X_test)
        else:
            model = train_random_forest(X_train.reshape(-1, 60), y_train)
            predictions = model.predict(X_test.reshape(-1, 60))
        
        # Inverse transform
        predictions = scaler.inverse_transform(predictions.reshape(-1,1))
        actual = scaler.inverse_transform(y_test.reshape(-1,1))
        
        # Prepare chart data
        dates = data.index[split+60:].strftime('%Y-%m-%d').tolist()
        
        return jsonify({
            'dates': dates,
            'actual': actual.flatten().tolist(),
            'predictions': predictions.flatten().tolist()
        })
    
    return render_template('index.html')