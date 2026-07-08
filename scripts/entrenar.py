import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from preparardatos import encode_target

# funcion para cargar el dataset preprocesado
def load_processed_data():
    """Carga el dataset preprocesado."""

    df = pd.read_csv("data/processed/train_processed.csv")

    return df

from utils import (
    split_features_target,
    split_train_test,
    scale_features,
    train_model,
    save_model
)

def main():

    df = load_processed_data()

    X, y = split_features_target(df)
    y = encode_target(y)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Escalar los datos
    X_train, X_test = scale_features(
    X_train,
    X_test,
    "models/scaler_v1.pkl"
)

    model = train_model(X_train, y_train)

    save_model(model, "models/model_v1.pkl")

    print("Entrenamiento:", X_train.shape)
    print("Prueba:", X_test.shape)

if __name__ == "__main__":
    main()