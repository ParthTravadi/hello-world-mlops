from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def home():
    return {"message": "Hello-World MLOps API", "endpoints": ["/predict"]}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        # Load model
        model_path = 'artifacts/model.pkl'
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found. Run 'python train.py' first"}), 404
            
        model = joblib.load(model_path)
        prediction = model.predict(features)[0]
        
        return jsonify({
            "prediction": int(prediction),
            "features": data['features']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
