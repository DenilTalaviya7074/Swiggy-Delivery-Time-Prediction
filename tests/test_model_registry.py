import mlflow
import pytest
from mlflow.tracking import MlflowClient
import dagshub

# Initialize Dagshub exactly like your friend's setup
dagshub.init(
    repo_owner="DenilTalaviya7074",
    repo_name="Swiggy-Delivery-Time-Prediction",
    mlflow=True
)
mlflow.set_tracking_uri("https://dagshub.com/DenilTalaviya7074/Swiggy-Delivery-Time-Prediction.mlflow")

def test_staging_model_loaded():
    client = MlflowClient()
    
    # THE FIX: Use the exact Registered Model Name from evaluation.py
    model_name = "delivery_time_model"
    stage = "Staging"
    
    try:
        # 1. Verify the model actually exists in the Staging stage
        latest_versions = client.get_latest_versions(name=model_name, stages=[stage])
        assert len(latest_versions) > 0, f"No versions found for model '{model_name}' in {stage}"
        
        # 2. Load the model using the classic URI format
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        assert model is not None, "Model loaded but is None"
        print(f"Successfully loaded {model_name} from {stage}!")
        
    except Exception as e:
        pytest.fail(f"Staging model could not be loaded: {e}")