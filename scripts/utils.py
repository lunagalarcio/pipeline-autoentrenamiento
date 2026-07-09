import json
import os
import joblib
import logging
from logging.handlers import RotatingFileHandler
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import re
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def get_logger(name):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    os.makedirs("logs", exist_ok=True)
    file_handler = RotatingFileHandler(
        "logs/pipeline.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(file_format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = get_logger(__name__)

# funcion para separar variables predictoras y variable objetivo
def split_features_target(df):

    X = df.drop("Churn", axis=1)

    y = df["Churn"]

    return X, y

# funcion para dividir el dataset en entrenamiento y prueba
def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test

# funcion para entrenar el modelo
def train_model(X_train, y_train):
    """Entrena el modelo."""

    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

# funcion para escalar las variables numericas
def scale_features(X_train, X_test, scaler_path):

    X_train = X_train.copy()
    X_test = X_test.copy()

    numerical_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    scaler = StandardScaler()

    X_train[numerical_columns] = scaler.fit_transform(
        X_train[numerical_columns]
    )

    X_test[numerical_columns] = scaler.transform(
        X_test[numerical_columns]
    )

    joblib.dump(scaler, scaler_path)

    logger.info(f"Scaler guardado en: {scaler_path}")

    return X_train, X_test

def save_model(model, version):

    path = f"models/model_{version}.pkl"

    joblib.dump(model, path)

    logger.info(f"Modelo guardado en {path}")
    
def get_latest_model():
    pattern = re.compile(r"model_v(\d+)_\d+T\d+\.pkl")
    model_files = []
    for file in os.listdir("models"):
        match = pattern.match(file)
        if match:
            model_files.append((int(match.group(1)), file))
    if not model_files:
        return None
    model_files.sort(key=lambda x: x[0], reverse=True)
    return os.path.join("models", model_files[0][1])

def get_latest_scaler():
    pattern = re.compile(r"scaler_v(\d+)_\d+T\d+\.pkl")
    scaler_files = []
    for file in os.listdir("models"):
        match = pattern.match(file)
        if match:
            scaler_files.append((int(match.group(1)), file))
    if not scaler_files:
        return None
    scaler_files.sort(key=lambda x: x[0], reverse=True)
    return os.path.join("models", scaler_files[0][1])

def save_scaler(scaler):

    filename = get_next_version(
        "models",
        "scaler"
    )

    path = f"models/{filename}.pkl"

    joblib.dump(scaler, path)

    logger.info(f"Scaler guardado en {path}")

    return path

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    logger.info("Métricas del modelo:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info(f"Matriz de confusión:\n{confusion_matrix(y_test, y_pred)}")

    return metrics
def create_version():
    os.makedirs("models", exist_ok=True)

    versions = []

    pattern = re.compile(r"model_v(\d+)")

    for file in os.listdir("models"):

        match = pattern.match(file)

        if match:
            versions.append(int(match.group(1)))

    next_version = 1 if not versions else max(versions) + 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"v{next_version}_{timestamp}"

def save_metrics(metrics, version):

    path = f"reports/metrics_{version}.json"

    with open(path, "w") as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    logger.info(f"Métricas guardadas en {path}")

def transform_features(X, scaler):
    X = X.copy()

    numerical_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    X[numerical_columns] = scaler.transform(
        X[numerical_columns]
    )

    return X

def get_next_version(folder, prefix):
    os.makedirs(folder, exist_ok=True)

    pattern = re.compile(rf"{prefix}_v(\d+)")

    versions = []

    for file in os.listdir(folder):

        match = pattern.match(file)

        if match:
            versions.append(int(match.group(1)))

    next_version = 1 if not versions else max(versions) + 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{prefix}_v{next_version}_{timestamp}"

    return filename

# funcion para archivar los nuevos datos
def archive_new_data():
    source_folder = "data/new"
    archive_folder = "data/archive"

    os.makedirs(archive_folder, exist_ok=True)

    for file in os.listdir(source_folder):

        if not file.endswith(".csv"):
            continue

        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(archive_folder, file)

        # Si ya existe un archivo con ese nombre
        if os.path.exists(destination_path):

            name, extension = os.path.splitext(file)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            destination_path = os.path.join(
                archive_folder,
                f"{name}_{timestamp}{extension}"
            )

        shutil.move(source_path, destination_path)

    logger.info("Archivos archivados correctamente.")