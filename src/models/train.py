import pandas as pd
import yaml
import joblib
import logging
import mlflow
import dagshub

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from pathlib import Path

TARGET = "time_taken"

# ================= LOGGER =================
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# ================= FUNCTIONS =================

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("File not found")
        raise

def read_params(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def save_model(model, save_dir: Path, model_name: str):
    joblib.dump(model, save_dir / model_name)

def save_transformer(transformer, save_dir: Path, name: str):
    joblib.dump(transformer, save_dir / name)

def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

# ================= MAIN =================

if __name__ == "__main__":

    # 👉 DAGSHUB + MLFLOW SETUP
    dagshub.init(
        repo_owner='DenilTalaviya7074',
        repo_name='Swiggy-Delivery-Time-Prediction',
        mlflow=True
    )

    mlflow.set_tracking_uri(
        "https://dagshub.com/DenilTalaviya7074/Swiggy-Delivery-Time-Prediction.mlflow"
    )

    # paths
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data" / "processed" / "train_trans.csv"
    params_path = root_path / "params.yaml"

    # load data
    df = load_data(data_path)
    logger.info("Data loaded")

    X_train, y_train = make_X_and_y(df, TARGET)

    params = read_params(params_path)["Train"]

    # models
    rf = RandomForestRegressor(**params["Random_Forest"])
    lgbm = LGBMRegressor(**params["LightGBM"])
    lr = LinearRegression()

    stacking = StackingRegressor(
        estimators=[("rf", rf), ("lgbm", lgbm)],
        final_estimator=lr,
        cv=5,
        n_jobs=-1
    )

    power_transform = PowerTransformer()

    model = TransformedTargetRegressor(
        regressor=stacking,
        transformer=power_transform
    )

    # ================= MLFLOW RUN =================
    with mlflow.start_run():

        # train
        model.fit(X_train, y_train)
        logger.info("Model trained")

        # 🔥 IMPORTANT: LOG MODEL TO MLFLOW
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="delivery_time_pred_model"
        )

        # log params
        mlflow.log_param("model_type", "stacking_regressor")

        # save locally (optional but good)
        model_dir = root_path / "models"
        model_dir.mkdir(exist_ok=True)

        save_model(model, model_dir, "model.joblib")
        save_model(model.regressor_, model_dir, "stacking_regressor.joblib")
        save_transformer(model.transformer_, model_dir, "power_transformer.joblib")

        logger.info("Model + transformer saved locally")
