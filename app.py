from flask import Flask, request, jsonify
import pickle

# Step 1: Load the trained model
with open("model/house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Step 2: Create a Flask app
app = Flask(__name__)

# Step 3: Define a route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Step 4: Get data from the request
    data = request.get_json()
    size = data.get("size")
    bedrooms = data.get("bedrooms")

    # Step 5: Make a prediction
    prediction = model.predict([[size, bedrooms]])

    # Step 6: Send back the prediction
    return jsonify({"predicted_price": prediction[0]})

# Step 7: Run the app
if __name__ == "__main__":
    app.run(debug=True)
