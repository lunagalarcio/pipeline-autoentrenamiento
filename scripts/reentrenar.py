import pandas as pd
from sklearn import metrics

# importar funciones que se usan en otros scripts
from utils import (
    evaluate_model,
    save_metrics,
    split_features_target,
    split_train_test,
    scale_features,
    train_model,
    save_model
)

from preparardatos import (
    clean_data,
    encode_features,
    encode_target
)

def load_old_data():
    """Carga el dataset utilizado para entrenar el modelo actual."""

    return pd.read_csv("data/processed/initial_train.csv")


def load_new_data():
    """Carga los nuevos datos que llegaron al sistema."""

    return pd.read_csv("data/new/new_customers.csv")

def merge_data(old_data, new_data):
    """Une los datos antiguos con los nuevos."""

    df = pd.concat(
        [old_data, new_data],
        ignore_index=True
    )

    print(f"Dataset combinado: {df.shape}")

    return df

def main():

    # Cargar datasets
    old_data = load_old_data()
    new_data = load_new_data()

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
    "models/scaler_v2.pkl")

    # Entrenar el nuevo modelo
    model = train_model(X_train, y_train)

    # Guardar la nueva versión del modelo
    save_model(model, "models/model_v2.pkl")

    # Evaluar el modelo
    metrics = evaluate_model(model, X_test, y_test)

    # Guardar las métricas
    save_metrics(metrics, "reports/metrics_v2.json")

if __name__ == "__main__":
    main()