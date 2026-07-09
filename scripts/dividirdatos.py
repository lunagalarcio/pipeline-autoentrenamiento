import pandas as pd
from pathlib import Path
from utils import get_logger

logger = get_logger(__name__)

# Crear las rutas
RAW_PATH = Path("data/raw/Telco-Customer-Churn.csv")
PROCESSED_PATH = Path("data/processed/initial_train.csv")
NEW_DATA_PATH = Path("data/new/new_customers.csv")

# Leer el dataset original
df = pd.read_csv(RAW_PATH)

# Calcular el 80% del dataset
division_index = int(len(df) * 0.8)

# Dividir el dataset entre entrenamiento inicial y nuevos datos
initial_data = df.iloc[:division_index]
new_data = df.iloc[division_index:]

# Guardar los archivos como csv en las carpetas correspondientes
initial_data.to_csv(PROCESSED_PATH, index=False)
new_data.to_csv(NEW_DATA_PATH, index=False)

logger.info(f"Dataset original: {len(df)} registros")
logger.info(f"Entrenamiento inicial: {len(initial_data)} registros")
logger.info(f"Nuevos datos: {len(new_data)} registros")