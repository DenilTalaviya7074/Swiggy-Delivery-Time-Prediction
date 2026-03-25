import pandas as pd
import requests
from pathlib import Path

# -----------------------------
# Path for data
# -----------------------------
root_path = Path(__file__).parent.parent
data_path = root_path / "data" / "raw" / "swiggy.csv"

# -----------------------------
# Prediction endpoint (LOCAL API)
# -----------------------------
predict_url = "http://127.0.0.1:8000/predict"

# -----------------------------
# Load sample row
# -----------------------------
sample_row = pd.read_csv(data_path).dropna().sample(1)

# Get actual target value
actual_value = sample_row.iloc[:, -1].values.item().replace("(min) ", "")
print(f"Actual delivery time: {actual_value} min")

# -----------------------------
# Prepare input data
# -----------------------------
data = sample_row.drop(columns=[sample_row.columns[-1]]).squeeze().to_dict()
print("Input data sent to API:")
print(data)

# -----------------------------
# Send request to API
# -----------------------------
response = requests.post(url=predict_url, json=data)

print("\nStatus code:", response.status_code)

# -----------------------------
# Handle response (FIXED)
# -----------------------------
if response.status_code == 200:
    result = response.json()   # ✅ FIX: handle JSON response
    predicted_value = result["predicted_delivery_time"]

    print(f"Predicted delivery time: {predicted_value:.2f} min")
else:
    print("Error:", response.status_code)