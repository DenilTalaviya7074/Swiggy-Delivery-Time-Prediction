import pytest
import mlflow
import dagshub
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Initialize DagsHub
dagshub.init(
    repo_owner="DenilTalaviya7074",
    repo_name="Swiggy-Delivery-Time-Prediction",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/DenilTalaviya7074/Swiggy-Delivery-Time-Prediction.mlflow"
)

# Utility functions
def load_model_information(file_path):
    with open(file_path) as f:
        return json.load(f)

def load_transformer(transformer_path):
    return joblib.load(transformer_path)

# Main test
def test_model_performance():

    #  root path
    root_path = Path(__file__).parent.parent

    # load model name
    file_path = root_path / "run_information.json"
    model_name = load_model_information(file_path)["model_name"]

    # use STAGING stage (since you fixed it)
    stage = "Staging"
    model_path = f"models:/{model_name}/{stage}"

    # load model from registry
    try:
        model = mlflow.sklearn.load_model(model_path)
    except Exception as e:
        pytest.fail(f"Model loading failed: {str(e)}")

    # load preprocessor
    preprocessor_path = root_path / "models" / "preprocessor.joblib"
    preprocessor = load_transformer(preprocessor_path)

    # create pipeline
    model_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", model)
    ])

    #  load test data
    test_data_path = root_path / "data" / "interim" / "test.csv"
    df = pd.read_csv(test_data_path)

    # clean
    df.dropna(inplace=True)

    # split
    X = df.drop(columns=["time_taken"])
    y = df["time_taken"]

    #  prediction
    y_pred = model_pipe.predict(X)

    #  evaluation
    mean_error = mean_absolute_error(y, y_pred)

    #  threshold check
    threshold_error = 5
    assert mean_error <= threshold_error, \
        f"Model failed: error {mean_error} > {threshold_error}"

    print(f"Average Error: {mean_error}")
    print(f"{model_name} passed performance test ")