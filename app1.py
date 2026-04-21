from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import sys
import os

app = Flask(__name__)

# Attempt to load the model on startup globally
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'career_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, 'model.pkl')
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {model_path}.")
except FileNotFoundError:
    print("WARNING: 'model.pkl' or 'career_model.pkl' not found in the root directory.", file=sys.stderr)
    model = None
except Exception as e:
    print(f"WARNING: An error occurred while loading the model: {e}", file=sys.stderr)
    model = None

@app.route('/')
def home():
    """Serves the main frontend UI."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to receive frontend data and return a prediction."""
    try:
        if not model:
            return jsonify({"status": "error", "message": "Model not loaded."}), 500

        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400

        # Extract 21 numerical features in the exact order
        num_features = [
            'grade1', 'grade2', 'final_grade', 'study_time', 'failures', 'absences',
            'openness', 'conscientiousness', 'extraversion', 'agreeableness',
            'neuroticism', 'coding_skill', 'communication_skill', 'analytical_skill',
            'study_hours', 'consistency', 'participation', 'tech_interest',
            'art_interest', 'business_interest', 'family_income'
        ]
        
        features_list = []
        for feat in num_features:
            features_list.append(float(data.get(feat, 0.0)))
            
        # Extract and one-hot encode the internet_access categorical feature
        # Expected features: 'internet_access_no', 'internet_access_yes'
        internet_access = str(data.get('internet_access', 'yes')).strip().lower()
        if internet_access == 'no':
            features_list.extend([1.0, 0.0])
        else:
            features_list.extend([0.0, 1.0])

        # Convert to 2D numpy array (1 sample, 23 features)
        features_array = np.array([features_list])

        # Pass the 2D array to the model
        prediction_result = model.predict(features_array)
        
        # Extract the string prediction
        predicted_career = str(prediction_result[0])

        # Return the exact requested JSON format
        return jsonify({
            "status": "success",
            "career": predicted_career
        })

    except Exception as e:
        # Catch any errors (e.g., parsing, model inference) and return the failure response
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    # Run the app on port 5001 with debug enabled
    app.run(port=5001, debug=True)
