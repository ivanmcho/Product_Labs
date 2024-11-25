import pandas as pd

# Lee el archivo Parquet desde una carpeta superior
df = pd.read_parquet("data/df_mlflow_test.parquet")

# Muestra las columnas del DataFrame

# Muestra las dimensiones (filas, columnas)
print(df.shape)
