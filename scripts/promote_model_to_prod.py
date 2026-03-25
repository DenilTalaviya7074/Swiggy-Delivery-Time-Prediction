import mlflow
import dagshub
import json
from mlflow import MlflowClient

dagshub.init(
    repo_owner='DenilTalaviya7074',
    repo_name='Swiggy-Delivery-Time-Prediction',
    mlflow=True
)

# set tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/DenilTalaviya7074/Swiggy-Delivery-Time-Prediction.mlflow"
)

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

# get model name
model_name = load_model_information("run_information.json")["model_name"]

stage = "Staging"
client = MlflowClient()

# get latest versions
latest_versions = client.get_latest_versions(name=model_name, stages=[stage])

# safety check
if not latest_versions:
    raise Exception("No model found in Staging stage")

latest_model_version_staging = latest_versions[0].version

# promote to production
client.transition_model_version_stage(
    name=model_name,
    version=latest_model_version_staging,
    stage="Production",
    archive_existing_versions=True
)

print("Model promoted to Production successfully!") 