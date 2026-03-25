import pytest
import mlflow
from mlflow import MlflowClient
import dagshub
import json
import os

dagshub.init(
    repo_owner="DenilTalaviya7074",
    repo_name="Swiggy-Delivery-Time-Prediction",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/DenilTalaviya7074/Swiggy-Delivery-Time-Prediction.mlflow"
)

def load_model_information(file_path):
    with open(file_path) as f:
        return json.load(f)

# correct path handling
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "run_information.json")

model_name = load_model_information(file_path)["model_name"]

@pytest.mark.parametrize("model_name, stage", [(model_name, "Staging")])
def test_load_model_from_registry(model_name, stage):
    client = MlflowClient()

    latest_versions = client.get_latest_versions(name=model_name, stages=[stage])

    # better assertion
    assert latest_versions, f"No versions found for model {model_name} in {stage}"

    latest_version = latest_versions[0].version

    model_path = f"models:/{model_name}/{stage}"

    try:
        # universal loader
        model = mlflow.pyfunc.load_model(model_path)
    except Exception as e:
        pytest.fail(f"Model loading failed: {str(e)}")

    assert model is not None, "Failed to load model from registry"

    print(f"The {model_name} model with version {latest_version} was loaded successfully")