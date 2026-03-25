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

# CHANGED: We now test for the alias "staging" instead of the stage "Staging"
@pytest.mark.parametrize("model_name, alias", [(model_name, "staging")])
def test_load_model_from_registry(model_name, alias):
    client = MlflowClient()

    try:
        # Fetch the exact model version assigned to this alias
        model_version_info = client.get_model_version_by_alias(name=model_name, alias=alias)
        latest_version = model_version_info.version
    except Exception as e:
        pytest.fail(f"No versions found for model '{model_name}' with alias '{alias}'. Error: {str(e)}")

    # The modern MLflow URI format for loading by alias is models:/model_name@alias
    model_path = f"models:/{model_name}@{alias}"

    try:
        # universal loader
        model = mlflow.pyfunc.load_model(model_path)
    except Exception as e:
        pytest.fail(f"Model loading failed: {str(e)}")

    assert model is not None, "Failed to load model from registry"

    print(f"The {model_name} model with version {latest_version} was loaded successfully")