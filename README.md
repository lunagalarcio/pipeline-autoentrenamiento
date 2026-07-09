# Pipeline de Autoentrenamiento

Pipeline automatizado para entrenamiento y reentrenamiento continuo de un modelo de clasificación de churn (Telco Customer Churn) usando **Python** y **GitHub Actions**.

## Flujo del Pipeline

1. **División de datos** — `scripts/dividirdatos.py` separa el dataset original en 80% entrenamiento inicial (`data/processed/initial_train.csv`) y 20% como simulación de nuevos datos (`data/new/new_customers.csv`).
2. **Pipeline automático** — `scripts/main.py` ejecuta la lógica principal:
   - Si **no existe un modelo** → Entrena modelo inicial (`scripts/entrenar.py`).
   - Si **hay nuevos datos** en `data/new/` → Reentrena el modelo (`scripts/reentrenar.py`).
   - Si **no hay cambios** → Mantiene el modelo actual.
3. **Entrenamiento** — Escalado de características numéricas, codificación one-hot, regresión logística. Guarda modelo (`.pkl`), scaler (`.pkl`) y métricas (`.json`) versionados.
4. **Reentrenamiento** — Fusiona datos viejos + nuevos, reentrena, versiona el nuevo modelo y archiva los datos nuevos en `data/archive/`.

## Estructura del Proyecto

```
├── .github/workflows/pipeline.yml   # Workflow de GitHub Actions
├── data/
│   ├── archive/                      # Datos nuevos procesados (archivados)
│   ├── new/                          # Nuevos datos entrantes (CSV)
│   ├── processed/                    # Datos procesados (train)
│   └── raw/                          # Dataset original
├── models/                           # Modelos y scalers entrenados (.pkl)
├── reports/                          # Métricas de evaluación (.json)
├── scripts/
│   ├── main.py                       # Orquestador del pipeline
│   ├── entrenar.py                   # Entrenamiento inicial
│   ├── reentrenar.py                 # Reentrenamiento con nuevos datos
│   ├── preparardatos.py              # Limpieza y preprocesamiento
│   ├── dividirdatos.py               # División inicial del dataset
│   ├── entenderdatos.py              # Análisis exploratorio
│   ├── evaluar.py                    # Evaluación de modelo
│   └── utils.py                      # Funciones reutilizables
├── dependencias.txt                  # Dependencias del proyecto
└── README.md
```

## GitHub Actions

El workflow `.github/workflows/pipeline.yml` se ejecuta automáticamente al hacer push a la rama `main` o manualmente desde la pestaña Actions. Pasos:

1. Clona el repositorio
2. Configura Python 3.13
3. Instala dependencias (`pip install -r dependencias.txt`)
4. Ejecuta `python scripts/main.py`

## Ejecución Local

```bash
# Instalar dependencias
pip install -r dependencias.txt

# Dividir datos iniciales (80% train, 20% nuevos)
python scripts/dividirdatos.py

# Ejecutar el pipeline
python scripts/main.py
```

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — Datos de clientes de una compañía de telecomunicaciones, utilizado para predecir si un cliente abandonará el servicio.

## Modelo

Regresión logística con `scikit-learn`. Cada versión genera un nuevo archivo con formato `model_v{version}_{timestamp}.pkl` y sus métricas asociadas en `reports/`.
