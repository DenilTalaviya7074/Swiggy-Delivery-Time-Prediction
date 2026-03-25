from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
import dagshub
from scripts.data_clean_utils import perform_data_cleaning

# -----------------------------
# Config
# -----------------------------
set_config(transform_output="pandas")

# -----------------------------
# DagsHub + MLflow setup
# -----------------------------
dagshub.init(
    repo_owner="DenilTalaviya7074",
    repo_name="Swiggy-Delivery-Time-Prediction",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/DenilTalaviya7074/Swiggy-Delivery-Time-Prediction.mlflow"
)

# -----------------------------
# Schema (RAW INPUT)
# -----------------------------
class Data(BaseModel):
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: float
    Delivery_person_Ratings: float
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: float
    Festival: str
    City: str

# -----------------------------
# Load model info
# -----------------------------
def load_model_information(file_path):
    with open(file_path) as f:
        return json.load(f)

model_name = load_model_information("run_information.json")["model_name"]

# -----------------------------
# Load model using latest version (FIXED)
# -----------------------------
client = MlflowClient()

latest_version = client.get_latest_versions(name=model_name)[0].version

model_uri = f"models:/{model_name}/{latest_version}"
model = mlflow.sklearn.load_model(model_uri)

# -----------------------------
# Load preprocessor
# -----------------------------
preprocessor = joblib.load("models/preprocessor.joblib")

# -----------------------------
# Create full pipeline
# -----------------------------
model_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", model)
])

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Swiggy Delivery Time Prediction API"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: Data):

    # Convert input to DataFrame
    pred_data = pd.DataFrame({
    'ID': data.ID,
    'Delivery_person_ID': data.Delivery_person_ID,
    'Delivery_person_Age': data.Delivery_person_Age,
    'Delivery_person_Ratings': data.Delivery_person_Ratings,
    'Restaurant_latitude': data.Restaurant_latitude,
    'Restaurant_longitude': data.Restaurant_longitude,
    'Delivery_location_latitude': data.Delivery_location_latitude,
    'Delivery_location_longitude': data.Delivery_location_longitude,
    'Order_Date': data.Order_Date,
    'Time_Orderd': data.Time_Orderd,
    'Time_Order_picked': data.Time_Order_picked,
    'Weatherconditions': data.Weatherconditions,
    'Road_traffic_density': data.Road_traffic_density,
    'Vehicle_condition': data.Vehicle_condition,
    'Type_of_order': data.Type_of_order,
    'Type_of_vehicle': data.Type_of_vehicle,
    'multiple_deliveries': data.multiple_deliveries,
    'Festival': data.Festival,
    'City': data.City}, index=[0])
    
    # # remove extra spaces from all string values
    # # remove extra spaces
    # pred_data = pred_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # # convert 'NaN' to actual NaN
    # import numpy as np
    # pred_data = pred_data.replace("NaN", np.nan)

    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data)

    # get predictions
    predictions = model_pipe.predict(cleaned_data)[0]

    return {
        "predicted_delivery_time": float(predictions)
    }

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("app:app")