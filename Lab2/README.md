# Universidad Galileo
# Mario Obed Morales Guitz
## 24006981

Este repositorio contiene el código y materiales utilizados en el **Laboratorio 2** del curso **Product Development**, parte del programa de **Maestría en Data Science** de la **Universidad Galileo**.

El laboratorio implementa un flujo de trabajo de **AutoML**, desplegado en un contenedor Docker, con dos modos de ejecución:
1. **Batch Prediction** (predicciones por lotes).
2. **API Prediction** (predicciones en tiempo real mediante API).

---

## Requisitos previos

Antes de comenzar, asegúrate de cumplir con los siguientes requisitos:
1. **Docker** instalado.
2. Directorio local `localpath/data` con los datos de entrada para el modelo.
3. Archivos `.env` configurados para cada modo de ejecución (detalles más adelante).

---

## Paso 1: Construir la imagen Docker

Ejecuta el siguiente comando en el directorio raíz del proyecto, donde se encuentra el `Dockerfile`:

```bash
docker build -t auto-ml:latest .
```

Esto crea la imagen Docker llamada `auto-ml:latest`.

---

## Paso 2: Ejecutar el contenedor

Puedes ejecutar el contenedor en dos modos: **Batch Prediction** o **API Prediction**. Sigue los pasos según el modo que prefieras.

---

### **Modo 1: Batch Prediction**

Este modo permite procesar datos por lotes, buscando automáticamente archivos nuevos en un intervalo de tiempo.

#### Pasos:
1. Asegúrate de tener un archivo `.env` llamado `batch_prediction.env` con las siguientes variables:
   ```env
   INPUT_FOLDER=/app/data/input
   OUTPUT_FOLDER=/app/data/output
   DATASET=data/dataset.parquet
   TARGET=Churn
   MODEL=NaiveBayes
   TRIALS=2
   ```
2. Crea los directorios `localpath/data/input` y `localpath/data/output` en tu máquina local.
3. Ejecuta el contenedor:
   ```bash
   docker run --env-file batch_prediction.env -v "$(pwd)/data":/app/data auto-ml:latest
   ```

**Explicación**:
- `--env-file batch_prediction.env`: Carga las variables de entorno necesarias.
- `-v "$(pwd)/data":/app/data`: Vincula tu carpeta local `data` al contenedor.
- `auto-ml:latest`: Usa la imagen Docker creada.

El contenedor procesará archivos en `/app/data/input` y generará resultados en `/app/data/output`.

---

### **Modo 2: API Prediction**

Este modo lanza un servidor de API para realizar predicciones en tiempo real.

#### Pasos:
1. Asegúrate de tener un archivo `.env` llamado `api_prediction.env` con las siguientes variables:
   ```env
   PORT=8000
   DATASET=data/dataset.parquet
   TARGET=Churn
   MODEL=GradientBoosting
   TRIALS=2
   ```
2. Ejecuta el contenedor:
   ```bash
   docker run --env-file api_prediction.env -v "$(pwd)/data":/app/data -p 8000:8000 auto-ml:latest
   ```

**Explicación**:
- `-p 8000:8000`: Expone el puerto 8000 para acceder a la API.
- `--env-file api_prediction.env`: Carga las variables de entorno necesarias.
- `-v "$(pwd)/data":/app/data`: Vincula tu carpeta local al contenedor.

Una vez ejecutado, la API estará disponible en `http://localhost:8000`.

---

## Paso 3: Usar la API (solo para API Prediction)

Con el contenedor en ejecución, puedes hacer predicciones enviando una solicitud `POST` a la API.

#### Ejemplo de solicitud con Python:
```python
import pandas as pd
import requests
import json

# Leer datos de prueba
df = pd.read_parquet('data/input/test_1.parquet')
data_json = df.to_dict(orient='records')

# URL de la API
url = 'http://localhost:8000/predict'

# Enviar solicitud POST
response = requests.post(url, json={"data": data_json})

# Mostrar resultados
print(json.dumps(response.json(), indent=2))
```

La API devolverá predicciones en formato JSON.

---

## Resumen de archivos `.env`

### `batch_prediction.env`
Ejemplo:
```env
INPUT_FOLDER=/app/data/input
OUTPUT_FOLDER=/app/data/output
DATASET=data/dataset.parquet
TARGET=Churn
MODEL=NaiveBayes
TRIALS=2
```

### `api_prediction.env`
Ejemplo:
```env
PORT=8000
DATASET=data/dataset.parquet
TARGET=Churn
MODEL=GradientBoosting
TRIALS=2
```

