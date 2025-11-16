from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

# Initialize the Flask app
app = Flask(__name__)

# Load the saved components (model, preprocessor, feature names) using robust relative paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'component' / 'models'

# Compatibility shim: some saved preprocessors reference a private
# class `_RemainderColsList` from older sklearn versions. When
# unpickling with a newer sklearn this name may be missing which
# raises `AttributeError`. Provide a minimal shim so unpickling can
# succeed. If you can, re-save your artifacts with the same sklearn
# version used for training to avoid these issues.
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, '_RemainderColsList'):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    # If sklearn internals change or import fails, we'll continue and
    # let the real error surface when loading artifacts.
    pass


def _load_joblib(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return joblib.load(path)

model = _load_joblib('best_growth_model.joblib')
preprocessor = _load_joblib('preprocessor.joblib')
feature_names = _load_joblib('feature_names.joblib')

# Define the inference function
def predict_growth_status(input_data):
    """
    Predicts the growth status for a single plant's data.
    
    :param input_data: A dictionary where keys are feature names and values are corresponding values.
    :return: A string indicating "Good Growth" or "Poor Growth"
    """
    try:
        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Apply preprocessing
        input_processed = preprocessor.transform(input_df)
        
        # Make the prediction
        prediction = model.predict(input_processed)
        
        # Return a human-readable result
        if prediction[0] == 1:
            return "Good Growth (Predicted: 1)"
        else:
            return "Poor Growth (Predicted: 0)"
    except Exception as e:
        return f"An error occurred during prediction: {e}"

# Define an endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Ensure the required features are in the input data
        required_features = set(feature_names)
        input_features = set(data.keys())
        
        if not required_features.issubset(input_features):
            return jsonify({"error": "Missing required features in input data"}), 400
        
        # Call the prediction function
        result = predict_growth_status(data)
        
        return jsonify({"prediction": result}), 200
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
