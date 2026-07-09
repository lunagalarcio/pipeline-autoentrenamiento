import pandas as pd
import os
import shutil
from datetime import datetime

from utils import (
    create_version,
    evaluate_model,
    save_metrics,
    split_features_target,
    split_train_test,
    scale_features,
    train_model,
    save_model,
    archive_new_data,
    get_logger
)

logger = get_logger(__name__)

# importar funciones de preparardatos.py
from preparardatos import (
    clean_data,
    encode_features,
    encode_target,
    validate_new_data
)

# funcion para cargar el dataset inicial
def load_old_data():
    return pd.read_csv("data/processed/initial_train.csv")

# funcion para cargar el dataset de nuevos clientes
def load_new_data():
    folder = "data/new"

    csv_files = [
        file
        for file in os.listdir(folder)
        if file.endswith(".csv")
    ]

    if not csv_files:
        return None

    dataframes = []

    for file in csv_files:

        path = os.path.join(folder, file)

        dataframes.append(
            pd.read_csv(path)
        )

    return pd.concat(
        dataframes,
        ignore_index=True
    )

# funcion para unir los datasets
def merge_data(old_data, new_data):
    df = pd.concat(
        [old_data, new_data],
        ignore_index=True
    )

    logger.info(f"Dataset combinado: {df.shape}")

    return df


def main():
# Crear una nueva versión para el modelo y las métricas
    version = create_version()

    new_data = load_new_data()

    if new_data is None:

        logger.info("No hay datos nuevos para reentrenar.")

        return

# Cargar datasets
    old_data = load_old_data()
    new_data = load_new_data()

# Validar calidad de los datos nuevos antes de reentrenar
    valido, errores = validate_new_data(new_data, old_data)
    if not valido:
        return

# Unir datos
    df = merge_data(old_data, new_data)

# Limpiar datos
    df = clean_data(df)

# Separar variables predictoras y objetivo
    X, y = split_features_target(df)
    y = encode_target(y)

# Codificar variables categóricas
    X = encode_features(X)

# Dividir entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Escalar variables numéricas
    X_train, X_test = scale_features(
    X_train,
    X_test,
    f"models/scaler_{version}.pkl")

# Entrenar el nuevo modelo
    model = train_model(X_train, y_train)

# Guardar la nueva versión del modelo
    save_model(model, version)

# Evaluar el modelo
    metrics = evaluate_model(model, X_test, y_test)

# Guardar las métricas
    save_metrics(metrics, version)

    archive_new_data()

if __name__ == "__main__":
    main()