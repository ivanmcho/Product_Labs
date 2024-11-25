import os
import logging
from dotenv import load_dotenv
from batch_prediction import run_batch
from api_prediction import start_prediction_api

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Punto de entrada principal del programa. Determina el tipo de despliegue
    (Batch o API) y ejecuta la lógica correspondiente.
    """
    # Cargar variables de entorno desde el archivo .env
    load_dotenv()
    deployment_type = os.getenv("DEPLOYMENT_TYPE")

    # Validar y ejecutar el tipo de despliegue
    if deployment_type == "Batch":
        logging.info("Ejecutando predicción en modo Batch.")
        run_batch()
    elif deployment_type == "API":
        logging.info("Ejecutando predicción en modo API.")
        start_prediction_api()
    else:
        logging.error("DEPLOYMENT_TYPE no válido. Debe ser 'Batch' o 'API'.")
        raise ValueError("DEPLOYMENT_TYPE no válido. Verifica tu configuración.")

if __name__ == "__main__":
    main()
