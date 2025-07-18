from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and preprocessor
model = pickle.load(open("artifacts/model.pkl", "rb"))
preprocessor = pickle.load(open("artifacts/preperocessor.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        coin = request.form['coin']
        symbol = request.form['symbol']
        volume = float(request.form['volume'])
        market_cap = float(request.form['market_cap'])
        return_1h = float(request.form['1h'])
        return_24h = float(request.form['24h'])
        return_7d = float(request.form['7d'])

        # Engineered features
        liquidity_ratio = volume / market_cap if market_cap != 0 else 0
        avg_return = np.mean([return_1h, return_24h, return_7d])
        volatility = (volume / market_cap) * avg_return if market_cap != 0 else 0

        input_dict = {
            'coin': coin,
            'symbol': symbol,
            'volume': volume,
            'market_cap': market_cap,
            '1h': return_1h,
            '24h': return_24h,
            '7d': return_7d,
            'liquidity_ratio': liquidity_ratio,
            'volatility': volatility
        }

        df = pd.DataFrame([input_dict])
        transformed = preprocessor.transform(df)
        prediction = model.predict(transformed)

        return render_template('index.html', prediction_text=f"Predicted Liquidity Level: {prediction[0]:.4f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

