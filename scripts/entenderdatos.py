import pandas as pd

# Cargar el dataset de entrenamiento
df = pd.read_csv("data/processed/initial_train.csv")

# info general
print(df.info())

# primeras filas
print(df.head())

# valores nulos
print(df.isnull().sum())

# estadisticas basicas
print(df.describe(include="all"))

# duplicados
print(df.duplicated().sum())

# valores unicos
print(df.nunique())

# tipos de datos
print(df.dtypes)