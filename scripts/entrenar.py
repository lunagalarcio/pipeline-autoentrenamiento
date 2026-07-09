import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from preparardatos import encode_target

# importar funciones de utils, reutilizables en varios scripts
from utils import (
    create_version,
    save_metrics,
    split_features_target,
    split_train_test,
    scale_features,
    train_model,
    save_model,
    evaluate_model
)

# funcion para cargar el dataset preprocesado
def load_processed_data():
    df = pd.read_csv("data/processed/train_processed.csv")
    return df

def main():
# Crear una nueva version para el modelo y el scaler
    version = create_version()
# Cargar el dataset preprocesado
    df = load_processed_data()
# Separar variables predictoras y variable objetivo
    X, y = split_features_target(df)
# Dividir el dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_train_test(X, y)
# Escalar los datos
    X_train, X_test = scale_features(
        X_train,
        X_test,
        f"models/scaler_{version}.pkl")
# Entrenar el modelo
    model = train_model(X_train, y_train)
# Guardar el modelo entrenado
    save_model(model, version)
# Evaluar el modelo y guardar las métricas
    metrics = evaluate_model( model, X_test, y_test)
    save_metrics(metrics, version)

if __name__ == "__main__":
    main()