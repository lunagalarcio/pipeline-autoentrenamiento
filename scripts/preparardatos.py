import pandas as pd
from utils import get_logger

logger = get_logger(__name__)

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
    logger.info(f"TotalCharges nulos tras limpieza: {df['TotalCharges'].isnull().sum()}")
    logger.info(f"Tipos de datos:\n{df.dtypes}")
    
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

    logger.info(f"Churn value counts:\n{df['Churn'].value_counts()}")
    logger.info(f"Churn proportions:\n{df['Churn'].value_counts(normalize=True)}")
    logger.info(f"Nulos por columna:\n{df.isnull().all()}")
    logger.info(f"Valores únicos por columna:\n{df.nunique()}")
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

    logger.info("Dataset procesado guardado correctamente.")

# Función para convertir la variable objetivo a valores binarios
def encode_target(y):
    if set(y.unique()) <= {0, 1}:
        return y

    return y.map({
        "No": 0,
        "Yes": 1
    })

# Función para validar calidad de datos nuevos antes de reentrenar
def validate_new_data(new_df, reference_df, max_null_pct=10):
    errores = []
    expected_cols = set(reference_df.columns)
    new_cols = set(new_df.columns)

    if new_cols != expected_cols:
        faltan = expected_cols - new_cols
        sobran = new_cols - expected_cols
        if faltan:
            errores.append(f"Faltan columnas: {faltan}")
        if sobran:
            errores.append(f"Columnas no esperadas: {sobran}")

    for col in new_df.columns:
        null_pct = new_df[col].isnull().mean() * 100
        if null_pct > max_null_pct:
            errores.append(f"Columna '{col}' tiene {null_pct:.1f}% de nulos (max {max_null_pct}%)")

    if "Churn" in new_df.columns:
        valores_esperados = {"Yes", "No", 0, 1, "0", "1"}
        valores_reales = set(new_df["Churn"].dropna().unique())
        if not valores_reales.issubset(valores_esperados):
            inesperados = valores_reales - valores_esperados
            errores.append(f"Columna 'Churn' tiene valores inesperados: {inesperados}")

    if "tenure" in new_df.columns:
        tenure_num = pd.to_numeric(new_df["tenure"], errors="coerce")
        negativos = (tenure_num < 0).sum()
        if negativos > 0:
            errores.append(f"Columna 'tenure' tiene {negativos} valores negativos")

    if "MonthlyCharges" in new_df.columns:
        mc_num = pd.to_numeric(new_df["MonthlyCharges"], errors="coerce")
        negativos = (mc_num < 0).sum()
        if negativos > 0:
            errores.append(f"Columna 'MonthlyCharges' tiene {negativos} valores negativos")

    if "TotalCharges" in new_df.columns:
        tc_num = pd.to_numeric(new_df["TotalCharges"], errors="coerce")
        negativos = (tc_num < 0).sum()
        if negativos > 0:
            errores.append(f"Columna 'TotalCharges' tiene {negativos} valores negativos")

    if errores:
        logger.error("=== VALIDACIÓN FALLIDA ===")
        for e in errores:
            logger.error(f"  - {e}")
        logger.error("Los datos nuevos no serán procesados.")
        return False, errores

    logger.info("Validación de datos nuevos: OK")
    return True, []


# Función principal para ejecutar el flujo de preprocesamiento
def preprocess_data():
    df = load_data()
    df = clean_data(df)

    # Separar variables predictoras y objetivo
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    y = encode_target(y)

    X = encode_features(X)
    logger.info(f"Shape después del encoding: {X.shape}")
    logger.info(f"Columnas: {list(X.columns)}")
    logger.info(f"Primeras filas:\n{X.head()}")

    save_processed_data(X, y)
    logger.info(f"Dataset procesado correctamente. Shape final: {X.shape}")


if __name__ == "__main__":
    preprocess_data()
    pass


