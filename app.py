from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("xgb_planet_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    if request.method == "POST":
        try:
            # Extract parameters from form
            params = [
                float(request.form["orb_period"]),
                float(request.form["tran_dur"]),
                float(request.form["tran_depth"]),
                float(request.form["planet_radius"]),
                float(request.form["eq_temp"]),
                float(request.form["insol_flux"]),
                float(request.form["stellar_teff"]),
                float(request.form["stellar_logg"]),
                float(request.form["stellar_rad"]),
                float(request.form["ra"]),
                float(request.form["dec"]),
            ]
            
            # Convert to array for prediction
            input_array = np.array([params])
            
            # Predict
            result = model.predict(input_array)[0]
            probability = model.predict_proba(input_array)[0][1]  # Probability of class 1
            
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result, probability=probability)

@app.route("/result", methods=["POST"])
def result():
    try:
        # Extract JSON payload
        data = request.get_json()

        # Ensure required keys exist
        required_fields = [
            "orb_period", "tran_dur", "tran_depth", "planet_radius",
            "eq_temp", "insol_flux", "stellar_teff", "stellar_logg",
            "stellar_rad", "ra", "dec"
        ]

        # Validate missing fields
        missing = [f for f in required_fields if f not in data]
        if missing:
            return {"error": f"Missing fields: {', '.join(missing)}"}, 400

        # Convert values to floats
        params = [float(data[field]) for field in required_fields]

        # Convert to numpy array for prediction
        input_array = np.array([params])

        # Predict
        result = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]  # Probability of class 1

        # ✅ Convert NumPy types to Python floats
        return {
            "result": float(result),
            "probability": float(probability)
        }

    except Exception as e:
        print("Error in /result:", e)
        return {"error": str(e)}, 500

@app.route("/batch_result", methods=["POST"])
def batch_result():
    try:
        # Extract JSON payload
        data = request.get_json()

        if not isinstance(data, list):
            return {"error": "Input must be a list of parameter objects"}, 400

        required_fields = [
            "orb_period", "tran_dur", "tran_depth", "planet_radius",
            "eq_temp", "insol_flux", "stellar_teff", "stellar_logg",
            "stellar_rad", "ra", "dec"
        ]

        results = []
        input_list = []

        # Validate and collect all parameter sets
        for i, item in enumerate(data):
            missing = [f for f in required_fields if f not in item]
            if missing:
                return {
                    "error": f"Missing fields in item {i}: {', '.join(missing)}"
                }, 400

            try:
                params = [float(item[field]) for field in required_fields]
                input_list.append(params)
            except ValueError as ve:
                return {
                    "error": f"Invalid numeric value in item {i}: {str(ve)}"
                }, 400

        # Convert to numpy array
        input_array = np.array(input_list)

        # Predict for all items at once
        preds = model.predict(input_array)
        probs = model.predict_proba(input_array)[:, 1]  # Probability of class 1

        # Build response
        for i in range(len(data)):
            results.append({
                "index": i,
                "result": float(preds[i]),
                "probability": float(probs[i])
            })

        return {"results": results}

    except Exception as e:
        print("Error in /batch_result:", e)
        return {"error": str(e)}, 500



if __name__ == "__main__":
    app.run(debug=True)
