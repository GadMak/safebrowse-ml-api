from flask import Flask, request, jsonify
import pickle
import numpy as np
from feature import FeatureExtraction
import os

app = Flask(__name__)

# Charger le modèle ML (chemin relatif depuis la racine du projet)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pickle', 'model.pkl')
with open(MODEL_PATH, 'rb') as file:
    gbc = pickle.load(file)

@app.route('/')
def home():
    return "API ML SafeBrowse OK"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "No url provided"}), 400

    try:
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, -1)
        y_pred = gbc.predict(x)[0]
        y_proba = gbc.predict_proba(x)[0].tolist()
        result = {
            "is_phishing": int(y_pred == -1),
            "proba_phishing": y_proba[0],
            "proba_safe": y_proba[1],
            "features": obj.getFeaturesList()
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Port dynamique pour Render
    app.run(host="0.0.0.0", port=port, debug=False)
