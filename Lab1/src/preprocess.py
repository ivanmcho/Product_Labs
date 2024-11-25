# src/preprocess.py
import pandas as pd
import sys
import yaml

def preprocess(input_file, output_file, features, target):
    # Cargar el dataset desde .parquet
    df = pd.read_parquet(input_file, engine='pyarrow')
    #print("Columnas en el archivo Parquet:", df.columns.tolist())  # Ver las columnas en el Parquet
    print(f'Se obtiene features: {features}')
    print(f'Se obtiene target: {target}')

    # Dividir las características y la variable objetivo
    selected_columns = features + [target]
    df_selected = df[selected_columns]

    # Guardar el dataset filtrado en CSV
    df_selected.to_csv(output_file, index=False, header=True)
    print(f"Extracción completa. Datos guardados en {output_file}")

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    params_file = sys.argv[3]

    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    preprocess(input_file, output_file, features, target)