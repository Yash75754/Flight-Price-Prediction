from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
import pandas as pd
import streamlit as st

app = Flask(__name__)
import pickle

with open("flight_rf.pkl", "rb") as file:
    model = pickle.load(file)



@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():

    date_dep = request.form["Dep_Time"]
    Journey_day = pd.to_datetime(date_dep).day
    Journey_month = pd.to_datetime(date_dep).month
    Dep_hour = pd.to_datetime(date_dep).hour
    Dep_min = pd.to_datetime(date_dep).minute

    date_arr = request.form["Arrival_Time"]
    Arrival_hour = pd.to_datetime(date_arr).hour
    Arrival_min = pd.to_datetime(date_arr).minute

    dur_hour = abs(Arrival_hour - Dep_hour)
    dur_min = abs(Arrival_min - Dep_min)

    Total_stops = int(request.form["stops"])

    airline = request.form["airline"]

    airline_dict = {
        "Jet Airways": [1,0,0,0,0,0,0,0,0,0],
        "IndiGo": [0,1,0,0,0,0,0,0,0,0],
        "Air India": [0,0,1,0,0,0,0,0,0,0],
        "Multiple carriers": [0,0,0,1,0,0,0,0,0,0],
        "SpiceJet": [0,0,0,0,1,0,0,0,0,0],
        "Vistara": [0,0,0,0,0,1,0,0,0,0],
        "GoAir": [0,0,0,0,0,0,1,0,0,0],
        "Multiple carriers Premium economy": [0,0,0,0,0,0,0,1,0,0],
        "Jet Airways Business": [0,0,0,0,0,0,0,0,1,0],
        "Vistara Premium economy": [0,0,0,0,0,0,0,0,0,1],
    }

    airline_features = airline_dict.get(airline, [0]*10)

    source = request.form["Source"]
    source_dict = {
        "Delhi": [1,0,0,0],
        "Kolkata": [0,1,0,0],
        "Mumbai": [0,0,1,0],
        "Chennai": [0,0,0,1]
    }
    s_Delhi, s_Kolkata, s_Mumbai, s_Chennai = source_dict.get(source, [0]*4)

    destination = request.form["Destination"]
    dest_dict = {
        "Cochin": [1,0,0,0,0],
        "Delhi": [0,1,0,0,0],
        "New_Delhi": [0,0,1,0,0],
        "Hyderabad": [0,0,0,1,0],
        "Kolkata": [0,0,0,0,1]
    }
    d_Cochin, d_Delhi, d_New_Delhi, d_Hyderabad, d_Kolkata = dest_dict.get(destination, [0]*5)

    prediction = model.predict([[
        Total_stops,
        Journey_day,
        Journey_month,
        Dep_hour,
        Dep_min,
        Arrival_hour,
        Arrival_min,
        dur_hour,
        dur_min,
        *airline_features,
        s_Chennai,
        s_Delhi,
        s_Kolkata,
        s_Mumbai,
        d_Cochin,
        d_Delhi,
        d_Hyderabad,
        d_Kolkata,
        d_New_Delhi
    ]])

    output = round(prediction[0], 2)

    return render_template("index.html", prediction_text=f"Your Flight price is Rs. {output}")


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
