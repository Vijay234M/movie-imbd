from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and encoders
model = joblib.load("movie_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        budget = float(request.form['budget'])
        popularity = float(request.form['popularity'])
        runtime = float(request.form['runtime'])
        release_year = int(request.form['release_year'])

        # Combine into feature array
        input_data = np.array([[budget, popularity, runtime, release_year]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        # If your label was encoded (optional):
        # genre_encoder = encoders['your_target_column_name']
        # result = genre_encoder.inverse_transform(prediction)[0]
        result = prediction[0]

        return render_template("index.html", prediction_text=f"Predicted Result: {result}")
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
