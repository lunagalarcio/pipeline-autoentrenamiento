import pandas as pd
from utils import get_logger

logger = get_logger(__name__)

df = pd.read_csv("data/processed/initial_train.csv")

logger.info(f"INFO del dataset:\n{df.info()}")
logger.info(f"Primeras filas:\n{df.head()}")
logger.info(f"Valores nulos:\n{df.isnull().sum()}")
logger.info(f"Estadísticas básicas:\n{df.describe(include='all')}")
logger.info(f"Duplicados: {df.duplicated().sum()}")
logger.info(f"Valores únicos:\n{df.nunique()}")
logger.info(f"Tipos de datos:\n{df.dtypes}")