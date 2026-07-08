import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# funcion para separar variables predictoras y variable objetivo
def split_features_target(df):
    """Separa variables predictoras y variable objetivo."""

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
    """Escala las variables numéricas y guarda el scaler."""

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

    print(f"Scaler guardado en: {scaler_path}")

    return X_train, X_test

def save_model(model, path):
    """Guarda el modelo entrenado."""

    joblib.dump(model, path)

    print(f"Modelo guardado en {path}")

    print("Modelo guardado correctamente.")

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo y devuelve sus métricas."""

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    print("\nMétricas del modelo")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nMatriz de confusión")
    print(confusion_matrix(y_test, y_pred))

    return metrics

def save_metrics(metrics, path):
    """Guarda las métricas del modelo."""

    os.makedirs("reports", exist_ok=True)

    with open(path, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Métricas guardadas en: {path}")

def transform_features(X, scaler):
    """Escala un conjunto de datos usando un scaler ya entrenado."""

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