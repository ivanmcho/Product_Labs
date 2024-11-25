import os
import logging
from dotenv import load_dotenv
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from model import train_and_optimize_model
import uvicorn

# Configuración del logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Definición del esquema de entrada para predicciones
class PredictionRequest(BaseModel):
    data: List[Dict]

# Inicialización de la API de predicciones
def start_prediction_api():
    app = FastAPI(
        title="Prediction API",
        description="API para realizar predicciones usando un modelo entrenado",
        version="1.0.0",
    )

    # # Entrenamiento del modelo al iniciar el servicio
    # logging.info("Iniciando el entrenamiento del modelo...")
    # trained_model = train_and_optimize_model()
    # logging.info("Modelo entrenado exitosamente.")

    # batch_prediction.py
    model_path = "models/trained_model.joblib"
    if os.path.exists(model_path):
        trained_model = joblib.load(model_path)
        logging.info("Modelo cargado correctamente.")
    else:
        logging.error("Modelo preentrenado no encontrado. Asegúrate de entrenarlo primero.")
        return

    # Ruta para verificar el estado de la API
    @app.get("/health", summary="Health Check", description="Verifica el estado del servicio")
    async def health_check():
        logging.info("Health check ejecutado correctamente.")
        return {"status": "ok"}

    # Ruta para realizar predicciones
    @app.post("/predict", summary="Realizar Predicción", description="Genera predicciones basadas en datos proporcionados")
    async def predict(input_data: PredictionRequest):
        logging.info("Solicitud de predicción recibida.")

        try:
            # Validar y procesar los datos de entrada
            input_dataframe = pd.DataFrame(input_data.data)

            if input_dataframe.empty:
                raise HTTPException(status_code=400, detail="El conjunto de datos enviado está vacío.")

            # Ruta del preprocesador
            preprocessor_path = "preprocessor.pkl"
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError("El archivo del preprocesador no se encontró en el sistema.")

            # Cargar el preprocesador
            preprocessor = joblib.load(preprocessor_path)
            processed_data = preprocessor.transform(input_dataframe)

            # Realizar las predicciones
            predictions = trained_model.predict_proba(processed_data)

            # Formatear las predicciones en un formato legible
            formatted_predictions = [
                {f"Class_{i+1}": prob for i, prob in enumerate(prob_row)}
                for prob_row in predictions
            ]

            logging.info("Predicciones generadas exitosamente.")
            return {"predictions": formatted_predictions}

        except Exception as error:
            logging.error(f"Error al generar predicciones: {str(error)}")
            raise HTTPException(status_code=500, detail=str(error))

    # Cargar variables de entorno
    load_dotenv()
    port = int(os.getenv("PORT", 8000))

    # Iniciar el servidor de la API
    logging.info("Iniciando el servidor de la API...")
    uvicorn.run(app, host="0.0.0.0", port=port)

# Punto de entrada del script
if __name__ == "__main__":
    start_prediction_api()
