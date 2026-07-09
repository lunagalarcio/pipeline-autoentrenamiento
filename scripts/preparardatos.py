import pandas as pd

# Funciones para preprocesamiento de datos
def load_data():
    df = pd.read_csv("data/processed/initial_train.csv")
    return df

# Función para limpiar y transformar los datos
def clean_data(df):
    # Eliminar columna customerID
    df = df.drop(columns=["customerID"])

    # Quitar espacios en blanco en nombres de columnas 
    df.columns = df.columns.str.strip()

    # Quitar espacios en blanco en valores de columnas de tipo string
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Convertir TotalCharges el cual esta como str a numérico
    df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce")

    # Para completar los valores faltantes, se revisasn si esos clientes son nuevos (tenure=0), TotalCharges = 0 
    mask_nuevos = df['TotalCharges'].isna() & (df['tenure'] == 0)
    df.loc[mask_nuevos, 'TotalCharges'] = 0.0

    # Si quedara algún NaN no explicado, se inserta con MonthlyCharges * tenure
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
    print(df["TotalCharges"].isnull().sum())
    print(df.dtypes)
    
    # Eliminar duplicados
    duplicados = df.duplicated().sum()
    df = df.drop_duplicates()

    # Verificar rangos válidos en columnas numéricas
    assert (df['tenure'] >= 0).all(), "Hay valores de tenure negativos"
    assert (df['MonthlyCharges'] >= 0).all(), "Hay MonthlyCharges negativos"
    assert (df['TotalCharges'] >= 0).all(), "Hay TotalCharges negativos"

    # Tipos de datos finales
    df['tenure'] = df['tenure'].astype(int)
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)

    # TotalCharges a dos decimales
    df['TotalCharges'] = df['TotalCharges'].round(2)

    # Verificar valores únicos en columnas categóricas
    print(df["Churn"].value_counts())
    # Verificar proporción de valores en la columna Churn
    print(df["Churn"].value_counts(normalize=True))

    # Verificar si hay valores nulos en todas las columnas
    print(df.isnull().all())

    # Verificar valores únicos en todas las columnas
    print(df.nunique())
    return df

# Función para separar variables predictoras y variable objetivo
def encode_features(X):
    X = pd.get_dummies(
        X,
        columns=X.select_dtypes(include="object").columns,
        drop_first=True,
        dtype=int
    )

    return X

# funcion para separar variables predictoras y objetivo
def save_processed_data(X, y):
    df_processed = X.copy()

    df_processed["Churn"] = y

    df_processed.to_csv(
        "data/processed/train_processed.csv",
        index=False
    )

    print("Dataset procesado guardado correctamente.")

# Función para convertir la variable objetivo a valores binarios
def encode_target(y):
    if set(y.unique()) <= {0, 1}:
        return y

    return y.map({
        "No": 0,
        "Yes": 1
    })

# Función principal para ejecutar el flujo de preprocesamiento
def preprocess_data():
    df = load_data()
    df = clean_data(df)

    # Separar variables predictoras y objetivo
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    y = encode_target(y)

    print(X.head())
    print(y.head())
    # Codificar las características categóricas
    X = encode_features(X)
    print("\nShape después del encoding:")
    print(X.shape)

    print("\nPrimeras columnas:")
    print(X.columns)

    print("\nPrimeras filas:")
    print(X.head())

    save_processed_data(X, y)
    print("Dataset procesado correctamente.")
    print("Shape final:", X.shape)


if __name__ == "__main__":
    preprocess_data()
    pass


