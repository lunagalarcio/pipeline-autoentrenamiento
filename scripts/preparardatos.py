# Este script divide los datos, para simular la llegada de nuevos clientes, y guarda los datasets resultantes en las carpetas correspondientes.
import pandas as pd
from pathlib import Path

# Crear las rutas
RAW_PATH = Path("data/raw/Telco-Customer-Churn.csv")
PROCESSED_PATH = Path("data/processed/initial_train.csv")
NEW_DATA_PATH = Path("data/new/new_customers.csv")

# Leer el dataset original
df = pd.read_csv(RAW_PATH)

# Calcular el 80%
division_index = int(len(df) * 0.8)

# Dividir el dataset
initial_data = df.iloc[:division_index]
new_data = df.iloc[division_index:]

# Guardar los archivos
initial_data.to_csv(PROCESSED_PATH, index=False)
new_data.to_csv(NEW_DATA_PATH, index=False)

print(f"Dataset original: {len(df)} registros")
print(f"Entrenamiento inicial: {len(initial_data)} registros")
print(f"Nuevos datos: {len(new_data)} registros")