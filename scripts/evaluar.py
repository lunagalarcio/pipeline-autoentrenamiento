import json
import joblib
import pandas as pd

from utils import (
    evaluate_model,
    save_metrics,
    transform_features,
    split_features_target,
    get_logger,
    split_features_target,
    split_train_test,
    scale_features 
)
# se crea un logger para registrar información durante la ejecución del script
logger = get_logger(__name__)

def load_data():
    df = pd.read_csv("data/processed/train_processed.csv")
    return df

def load_model():
    model = joblib.load("models/model_v1.pkl")
    return model

def load_scaler(path):
    """Carga un scaler guardado."""
    return joblib.load(path)

def main():

    df = load_data()

    X, y = split_features_target(df)

    _, X_test, _, y_test = split_train_test(X, y)

    scaler = load_scaler("models/scaler_v1.pkl")

    X_test = transform_features(X_test, scaler)
    model = load_model()

    logger.info("Modelo cargado correctamente.")
    logger.info(f"X_test head:\n{X_test.head()}")

    y_pred = predict(model, X_test)

    metrics = evaluate_model(model, X_test, y_test)

    save_metrics(metrics, "reports/metrics_v1.json")

if __name__ == "__main__":
    main()