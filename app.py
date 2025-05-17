from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model and label encoder once when app starts
model = pickle.load(open('xgb_model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features string from form, split and filter empty strings
        raw_features = request.form['features']
        features = [float(x.strip()) for x in raw_features.split(',') if x.strip() != '']

        # Validate feature length
        if len(features) != 41:
            raise ValueError("Input must contain exactly 41 numeric features.")

        # Prepare input and make prediction
        input_data = np.array(features).reshape(1, -1)
        pred = model.predict(input_data)[0]

        # Get confidence (max predicted probability)
        conf = max(model.predict_proba(input_data)[0]) * 100

        # Convert numeric prediction to original label
        pred_label = label_encoder.inverse_transform([pred])[0]

        # Return prediction text with label and confidence
        return render_template('index.html', prediction_text=f'Prediction: {pred_label} (Confidence: {conf:.1f}%)')

    except Exception as e:
        # Return error message if any exception occurs
        return render_template('index.html', prediction_text=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
