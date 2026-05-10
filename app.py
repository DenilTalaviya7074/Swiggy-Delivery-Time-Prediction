import asyncio
from pathlib import Path
from urllib.parse import parse_qs
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import joblib
import time
import random
import math
import requests
from sklearn import set_config
import dagshub
from scripts.data_clean_utils import perform_data_cleaning
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load the hidden variables from your .env file
load_dotenv()


# Tell FastAPI where your HTML file is
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

set_config(transform_output='pandas')

dagshub.init(repo_owner='DenilTalaviya7074', repo_name='Swiggy-Delivery-Time-Prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/DenilTalaviya7074/Swiggy-Delivery-Time-Prediction.mlflow")

def load_transformer(transformer_path):
    return joblib.load(transformer_path)

model_name = "delivery_time_model"
stage = "Production"
model_path = f"models:/{model_name}/{stage}"

max_retries = 3
model = None

for attempt in range(max_retries):
    try:
        model = mlflow.sklearn.load_model(model_path)
        print("Production model loaded successfully!")
        break
    except Exception as e:
        if attempt < max_retries - 1: time.sleep(5)
        else: raise e

preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)
model_pipe = Pipeline(steps=[('preprocess', preprocessor), ("regressor", model)])

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def root(request: Request): return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request): return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request): return RedirectResponse(url="/dashboard", status_code=303)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request): return templates.TemplateResponse("index.html", {"request": request})

@app.get("/result", response_class=HTMLResponse)
async def result_page(request: Request): return templates.TemplateResponse("result.html", {"request": request})

class PredictionRequest(BaseModel):
    Operating_City: str  
    Delivery_person_Age: float
    Delivery_person_Ratings: float
    Weatherconditions: str
    Road_traffic_density: str
    Type_of_order: str
    Type_of_vehicle: str
    Vehicle_condition: int
    multiple_deliveries: float
    Festival: str
    City: str
    Distance: float

def get_live_environment(city_name, default_weather, default_traffic):
    # Securely fetch the key from the .env file!
    API_KEY = os.getenv("OPENWEATHER_API_KEY") 
    
    weather_condition = default_weather
    
    if API_KEY: # Now it checks if the key was successfully loaded
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}"
            res = requests.get(url).json()
            
            if str(res.get("cod")) != "200":
                print(f"[ERROR] API Failed: {res.get('message')}")
            else:
                main_weather = res['weather'][0]['main'].lower()
                print(f"[DEBUG] Live weather for {city_name}: {main_weather}")
                
                # Enhanced mapping
                if "cloud" in main_weather: weather_condition = "conditions Cloudy"
                elif "rain" in main_weather or "storm" in main_weather: weather_condition = "conditions Stormy"
                elif "mist" in main_weather or "fog" in main_weather or "haze" in main_weather or "smoke" in main_weather or "dust" in main_weather: 
                    weather_condition = "conditions Fog"
                elif "clear" in main_weather: weather_condition = "conditions Sunny"
                
        except Exception as e: 
            print(f"[CRITICAL ERROR] Python failed to process API: {e}")

    # Time-Based Live Traffic Simulation
    hour = datetime.now().hour
    if 7 <= hour <= 10 or 17 <= hour <= 20: traffic = "Jam "
    elif 11 <= hour <= 16: traffic = "Medium "
    elif 21 <= hour <= 23: traffic = "Low "
    else: traffic = default_traffic

    return weather_condition, traffic

def calculate_financials(distance, eta, vehicle_type):
    # 1. Calculate Revenue
    revenue = 40 + (distance * 12)

    # 2. Calculate Fuel Cost based on Vehicle
    fuel_rates = {"motorcycle ": 2.5, "scooter ": 2.0, "electric_scooter ": 0.5, "bicycle ": 0.0}
    fuel_cost = distance * fuel_rates.get(vehicle_type.lower(), 2.0)

    # 3. Calculate Driver Pay (Base + Time)
    driver_pay = 20 + (eta * 2)

    # 4. Calculate Final Profit
    total_cost = fuel_cost + driver_pay
    profit = revenue - total_cost
    margin = (profit / revenue) * 100 if revenue > 0 else 0

    return {
        "revenue": round(revenue, 2),
        "cost": round(total_cost, 2),
        "profit": round(profit, 2),
        "margin": round(margin, 1)
    }

@app.post("/predict")
async def predict_eta(req: PredictionRequest):
    try:
        now = datetime.now()
        live_weather, live_traffic = get_live_environment(req.Operating_City.strip(), req.Weatherconditions, req.Road_traffic_density)

        # --- UPDATED NATIONWIDE COORDINATES ---
        city_coords = {
            # Gujarat
            "Ahmedabad": (23.0350, 72.5293), "Surat": (21.1702, 72.8311),
            "Vadodara": (22.3072, 73.1812), "Rajkot": (22.3039, 70.8022),
            "Gandhinagar": (23.2156, 72.6369),
            # Maharashtra
            "Mumbai": (19.0760, 72.8777), "Pune": (18.5204, 73.8567),
            # Delhi
            "Delhi": (28.7041, 77.1025),
            # Karnataka
            "Bangalore": (12.9716, 77.5946),
            # Telangana
            "Hyderabad": (17.3850, 78.4867),
            # Tamil Nadu
            "Chennai": (13.0827, 80.2707),
            # West Bengal
            "Kolkata": (22.5726, 88.3639)
        }
        
        center_lat, center_lon = city_coords.get(req.Operating_City.strip(), (28.7041, 77.1025))

        # Start Point (Restaurant)
        base_lat = center_lat + random.uniform(-0.02, 0.02)
        base_lon = center_lon + random.uniform(-0.02, 0.02)
        
        # MULTI-STOP ROUTING LOGIC
        num_deliveries = max(1, int(req.multiple_deliveries) + 1)
        customers = []
        segment_distance = req.Distance / num_deliveries
        current_lat, current_lon = base_lat, base_lon

        for _ in range(num_deliveries):
            bearing = random.uniform(0, 2 * math.pi)
            lat_change = (segment_distance * math.cos(bearing)) / 111.0
            lon_change = (segment_distance * math.sin(bearing)) / (111.0 * math.cos(math.radians(current_lat)))
            current_lat += lat_change
            current_lon += lon_change
            customers.append([current_lat, current_lon])

        model_input = {
            'ID': '0xUI_TEST', 'Delivery_person_ID': 'UI_USER_01',
            'Delivery_person_Age': req.Delivery_person_Age, 'Delivery_person_Ratings': req.Delivery_person_Ratings,
            'Restaurant_latitude': base_lat, 'Restaurant_longitude': base_lon,
            'Delivery_location_latitude': customers[-1][0], 'Delivery_location_longitude': customers[-1][1],
            'Order_Date': now.strftime("%d-%m-%Y"), 'Time_Orderd': now.strftime("%H:%M:%S"),
            'Time_Order_picked': (now + timedelta(minutes=15)).strftime("%H:%M:%S"),
            'Weatherconditions': live_weather, 'Road_traffic_density': live_traffic,
            'Vehicle_condition': req.Vehicle_condition, 'Type_of_order': req.Type_of_order,
            'Type_of_vehicle': req.Type_of_vehicle, 'multiple_deliveries': req.multiple_deliveries,
            'Festival': req.Festival, 'City': req.City
        }

        df = pd.DataFrame([model_input])
        cleaned_df = perform_data_cleaning(df)
        prediction = model_pipe.predict(cleaned_df)[0]

        return {
        "prediction": float(prediction),
        "environment": {"weather": live_weather, "traffic": live_traffic},
        "route": {
            "restaurant": [base_lat, base_lon],
            "customers": customers 
        },
        # YOU MISSED THIS LINE RIGHT HERE! 
        "financials": calculate_financials(req.Distance, float(prediction), req.Type_of_vehicle)
    }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)