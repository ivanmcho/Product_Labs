# src/evaluate.py
import pandas as pd
import joblib
import json
import sys
from sklearn.metrics import mean_squared_error, r2_score
import yaml

def evaluate(input_file, model_file, metrics_file, params_file):
    # Cargar el dataset de prueba desde .csv
    df = pd.read_csv(input_file)

    # Leer los parámetros
    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']
    
    # Cargar el preprocesador y transformar las características
    # preprocessor = joblib.load("preprocessor.joblib")
    # X = preprocessor.transform(df[features])
    # y = df[target]


    X = df[features]
    y = df[target]

    # Cargar el modelo entrenado
    model = joblib.load(model_file)

    # Realizar predicciones
    predictions = model.predict(X)

    # Calcular métricas
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    # Guardar métricas en un archivo JSON
    with open(metrics_file, 'w') as f:
        json.dump({'mse': mse, 'r2': r2}, f, indent=4)
    print(f"Métricas guardadas en {metrics_file}")

if __name__ == "__main__":
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    metrics_file = sys.argv[3]
    params_file = sys.argv[4]

    evaluate(input_file, model_file, metrics_file, params_file)
