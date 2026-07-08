import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# funcion para cargar el dataset preprocesado
def load_processed_data():
    """Carga el dataset preprocesado."""

    df = pd.read_csv("data/processed/train_processed.csv")

    return df

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
def scale_features(X_train, X_test):
    """Escala las variables numéricas."""

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

    joblib.dump(scaler, "models/scaler.pkl")

    return X_train, X_test

def save_model(model):
    """Guarda el modelo entrenado."""

    joblib.dump(
        model,
        "models/model_v1.pkl"
    )

    print("Modelo guardado correctamente.")

def main():

    df = load_processed_data()

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Escalar los datos
    X_train, X_test = scale_features(X_train, X_test)

    model = train_model(X_train, y_train)

    save_model(model)

    print("Entrenamiento:", X_train.shape)
    print("Prueba:", X_test.shape)

if __name__ == "__main__":
    main()