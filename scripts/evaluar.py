import joblib
import pandas as pd

from utils import (
    evaluate_model,
    save_metrics,
    transform_features,
    split_features_target,
    split_train_test,
    get_latest_model,
    get_latest_scaler,
    get_logger
)

logger = get_logger(__name__)


def main():
    df = pd.read_csv("data/processed/train_processed.csv")

    X, y = split_features_target(df)

    _, X_test, _, y_test = split_train_test(X, y)

    model_path = get_latest_model()
    scaler_path = get_latest_scaler()

    if model_path is None or scaler_path is None:
        logger.error("No hay modelo o scaler entrenado. Ejecuta primero el pipeline.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_test = transform_features(X_test, scaler)

    logger.info(f"Modelo cargado: {model_path}")
    logger.info(f"Scaler cargado: {scaler_path}")

    metrics = evaluate_model(model, X_test, y_test)

    version = model_path.replace("models/model_", "").replace(".pkl", "")
    save_metrics(metrics, version)


if __name__ == "__main__":
    main()