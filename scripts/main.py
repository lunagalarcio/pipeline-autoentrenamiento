import os
import subprocess
from utils import get_logger

logger = get_logger(__name__)

def model_exists():
    folder = "models"

    if not os.path.exists(folder):
        return False

    return any(
        file.startswith("model_") and file.endswith(".pkl")
        for file in os.listdir(folder)
    )

def new_data_exists():
    folder = "data/new"

    if not os.path.exists(folder):
        return False

    return any(
        file.endswith(".csv")
        for file in os.listdir(folder)
    )

def train():
    subprocess.run(
        ["python", "scripts/entrenar.py"],
         check=True
    )

def retrain():
    subprocess.run(
        ["python", "scripts/reentrenar.py"],
         check=True
    )

def main():

    if not model_exists():

        logger.info("No existe un modelo entrenado. Entrenando modelo inicial...")

        train()

    elif new_data_exists():

        logger.info("Se detectaron nuevos datos. Reentrenando modelo...")

        retrain()

    else:

        logger.info("No hay nuevos datos. Se mantiene el modelo actual.")


if __name__ == "__main__":
    main()