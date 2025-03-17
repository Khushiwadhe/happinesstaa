from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load dataset
df = pd.read_csv("India_Travel_Dataset.csv")

# Load trained model and encoders
with open("travel_recommendation_model.pkl", "rb") as model_file:
    knn = pickle.load(model_file)

with open("label_encoders.pkl", "rb") as encoder_file:
    label_encoders = pickle.load(encoder_file)

# Extract unique zones, states, and cities dynamically from the dataset
zones = df["Zone"].unique().tolist()
states_by_zone = df.groupby("Zone")["State"].unique().apply(list).to_dict()
cities_by_state = df.groupby("State")["City"].unique().apply(list).to_dict()


@app.route("/")
def home():
    return render_template("index.html")  # Home page


@app.route("/recommendation")
def recommendation():
    return render_template("recommendation.html", zones=zones)  # User input form


@app.route("/get_states", methods=["POST"])
def get_states():
    """Returns states based on the selected zone from the dataset"""
    zone = request.json["zone"]
    states = states_by_zone.get(zone, [])
    return jsonify(states)


@app.route("/get_cities", methods=["POST"])
def get_cities():
    """Returns cities based on the selected state from the dataset"""
    state = request.json["state"]
    cities = cities_by_state.get(state, [])
    return jsonify(cities)


@app.route("/recommend", methods=["POST"])
def recommend():
    """Process user input and recommend travel destinations"""
    data = request.form
    user_input = [
        label_encoders["Zone"].transform([data["zone"]])[0],
        label_encoders["State"].transform([data["state"]])[0],
        label_encoders["City"].transform([data["city"]])[0],
        label_encoders["Age Group"].transform([data["age_group"]])[0],
        label_encoders["Gender"].transform([data["gender"]])[0],
        label_encoders["Ideal Travel Months"].transform([data["ideal_months"]])[0],
        label_encoders["Budget"].transform([data["budget"]])[0]
    ]

    # Get nearest recommendations
    distances, indices = knn.kneighbors([user_input])
    recommendations = df.iloc[indices[0]][["Category", "Destination"]].to_dict(orient="records")

    return render_template("recommendations.html", recommendations=recommendations)  # Display travel suggestions


if __name__ == "__main__":
    app.run(debug=True)

