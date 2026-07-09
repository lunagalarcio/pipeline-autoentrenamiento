import os
import subprocess

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

        print("No existe un modelo entrenado.")
        print("Entrenando modelo inicial...")

        train()

    elif new_data_exists():

        print("Se detectaron nuevos datos.")
        print("Reentrenando modelo...")

        retrain()

    else:

        print("No hay nuevos datos.")
        print("Se mantiene el modelo actual.")


if __name__ == "__main__":
    main()