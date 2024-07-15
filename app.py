# Load the model
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
model = pickle.load(open('MedInsuranceCostModel.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input data
        sex = request.form.get("sex")
        bmi = float(request.form.get("bmi"))  # Assuming BMI is a numerical value
        age = int(request.form.get("age"))  # Assuming age is an integer
        children = int(request.form.get("children"))  # Assuming children is an integer
        region = request.form.get("region")
        smoker = request.form.get("smoker")  # Will be "yes" or "no"

        # Preprocess data (convert categorical variables)
        sex = 1 if sex == "male" else 0
        smoker = 1 if smoker == "yes" else 0
        region_mapping = {"nyanza": 0, "riftvalley": 1, "coast": 2, "northeastearn": 3}
        region = region_mapping.get(region.lower(), -1)

        if region == -1:
            return render_template("index.html", prediction="Invalid region provided")

        # Combine all features into a single numpy array
        preprocessed_data = np.array([[age, sex, bmi, children, smoker, region]])

        # Make prediction using your model's specific prediction method
        prediction = model.predict(preprocessed_data)[0]  # Adjust if your model returns a different format

        return render_template("index.html", prediction=f"Estimated Insurance Cost: ${prediction:.2f}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
