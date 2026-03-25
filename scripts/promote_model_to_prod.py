import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import warnings

# Ignore MLflow deprecation warnings for cleaner logs
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

# Initialize DagsHub
dagshub.init(
    repo_owner="DenilTalaviya7074",
    repo_name="Swiggy-Delivery-Time-Prediction",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/DenilTalaviya7074/Swiggy-Delivery-Time-Prediction.mlflow"
)

if __name__ == "__main__":
    client = MlflowClient()

    #THE FIX: Use the actual registered model name
    model_name = "delivery_time_model"
    from_stage = "Staging"
    to_stage = "Production"

    try:
        # 1. Fetch the model currently in Staging
        latest_versions = client.get_latest_versions(name=model_name, stages=[from_stage])
        
        if not latest_versions:
            raise Exception(f"No model found in {from_stage} stage for '{model_name}'")
        
        latest_version = latest_versions[0].version
        print(f"Found '{model_name}' version {latest_version} in {from_stage}.")
        
        # 2. Promote that specific version to Production
        print(f"Promoting to {to_stage}...")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage=to_stage,
            archive_existing_versions=True # Archives any older Production models
        )
        
        print(f"Model '{model_name}' version {latest_version} successfully promoted to {to_stage}!")
        
    except Exception as e:
        print(f"Error during promotion: {str(e)}")
        raise e