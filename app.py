from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import joblib
import time
from sklearn import set_config
import dagshub
from scripts.data_clean_utils import perform_data_cleaning

# set the output as pandas
set_config(transform_output='pandas')

# initialize dagshub (Updated to your repository)
dagshub.init(
    repo_owner='DenilTalaviya7074', 
    repo_name='Swiggy-Delivery-Time-Prediction', 
    mlflow=True
)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/DenilTalaviya7074/Swiggy-Delivery-Time-Prediction.mlflow")

# Schema
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

def load_transformer(transformer_path):
    return joblib.load(transformer_path)

# -----------------------------
# SAFE MODEL LOADING
# -----------------------------
# 1. Hardcode the exact name of the model registered in your DagsHub
model_name = "delivery_time_model"
stage = "Production"
model_path = f"models:/{model_name}/{stage}"

# 2. Network-safe retry loop to prevent 'IncompleteRead' crashes
max_retries = 3
model = None

for attempt in range(max_retries):
    try:
        print(f"Fetching '{model_name}' from {stage}...")
        model = mlflow.sklearn.load_model(model_path)
        print("Production model loaded successfully!")
        break
    except Exception as e:
        print(f"Network drop detected on attempt {attempt + 1}: {e}")
        if attempt < max_retries - 1:
            print("Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("Failed to load Production model.")
            raise e

# load the preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ("regressor", model)
])

# create the app
app = FastAPI()

# create the home endpoint
@app.get(path="/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"

# create the predict endpoint
@app.post(path="/predict")
def do_predictions(data: Data):
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
        'City': data.City
        }, index=[0]
    )
    
    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data)
    
    # get the predictions
    predictions = model_pipe.predict(cleaned_data)[0]

    return {"predicted_delivery_time": float(predictions)}

if __name__ == "__main__":
    # FIX: Pass 'app' directly instead of "app:app" to stop the double-download crash
    uvicorn.run(app='app:app',host="0.0.0.0", port=8000)