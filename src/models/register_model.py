import mlflow
import dagshub
from mlflow.tracking import MlflowClient
import logging

# ================= LOGGER =================
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# ================= MAIN =================
if __name__ == "__main__":

    # 1. Initialize Dagshub and MLflow tracking
    dagshub_uri = "https://dagshub.com/DenilTalaviya7074/Swiggy-Delivery-Time-Prediction.mlflow"
    
    dagshub.init(
        repo_owner='DenilTalaviya7074',
        repo_name='Swiggy-Delivery-Time-Prediction',
        mlflow=True
    )
    mlflow.set_tracking_uri(dagshub_uri)
    
    # Initialize the client with the exact URI to prevent local path confusion
    client = MlflowClient(tracking_uri=dagshub_uri)

    # 2. Use the exact name that evaluation.py already registered the model under!
    model_name = "delivery_time_model" 

    logger.info(f"Looking for registered model: {model_name}")

    # 3. Grab the latest registered version
    try:
        # get_latest_versions returns a list, so we grab the first item's version
        latest_versions = client.get_latest_versions(name=model_name)
        latest_version = latest_versions[0].version
        logger.info(f"Found existing version: {latest_version}")
    except Exception as e:
        logger.error(f"Could not find model '{model_name}'. Did evaluation.py run successfully?")
        raise e

    # 4. Transition the found version to Staging
    logger.info("Transitioning model to Staging...")
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Staging",
        archive_existing_versions=True  # Automatically moves older Staging models to Archived
    )

    logger.info(f"Successfully transitioned version {latest_version} of '{model_name}' to Staging!")