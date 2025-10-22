from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__)

model = tf.keras.models.load_model('gru_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']
        features = np.array([data])
        features_scaled = scaler.transform(features)
        features_scaled = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
        prediction = model.predict(features_scaled)
        return jsonify({'fraud_probability': float(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
