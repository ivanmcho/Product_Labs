import os
import time
import shutil
import pandas as pd
from model import train_and_optimize_model
import joblib
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_batch():
    """
    Función principal para ejecutar el proceso de modelamiento batch.
    Monitorea un directorio en busca de archivos nuevos, los procesa y genera predicciones.
    """
    logging.info("Inicio del proceso de modelamiento batch.")

    # Carga el modelo entrenado
    # try:
    #     model = train_and_optimize_model()
    # except Exception as e:
    #     logging.error(f"Error al cargar el modelo: {e}")
    #     return

    # batch_prediction.py
    model_path = "models/trained_model.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logging.info("Modelo cargado correctamente.")
    else:
        logging.error("Modelo preentrenado no encontrado. Asegúrate de entrenarlo primero.")
        return


    # Configuración de carpetas
    input_folder = os.getenv("INPUT_FOLDER")
    output_folder = os.getenv("OUTPUT_FOLDER")
    processed_folder = os.path.join(input_folder, 'processed')
    os.makedirs(processed_folder, exist_ok=True)

    logging.info(f"Carpeta de entrada configurada: {input_folder}")
    logging.info(f"Carpeta de salida configurada: {output_folder}")
    logging.info(f"Esperando archivos en: {input_folder}")

    # Parámetros para manejo de inactividad
    inactivity_counter = 0
    max_inactivity_cycles = 6
    inactivity_wait_time = 10  # Segundos de espera entre ciclos

    while True:
        # Filtrar archivos nuevos con extensión .parquet
        new_files = [file for file in os.listdir(input_folder) if file.endswith('.parquet')]

        if not new_files:
            logging.warning("No se encontraron archivos nuevos. Esperando...")
            inactivity_counter += 1
            if inactivity_counter >= max_inactivity_cycles:
                logging.warning("Inactividad prolongada. Finalizando proceso.")
                break
            time.sleep(inactivity_wait_time)
            continue

        # Reinicia el contador de inactividad al encontrar archivos
        inactivity_counter = 0

        for file_name in new_files:
            try:
                logging.info(f"Procesando archivo: {file_name}")

                # Cargar el archivo de entrada
                input_path = os.path.join(input_folder, file_name)
                input_data = pd.read_parquet(input_path)

                # Cargar el preprocesador entrenado
                preprocessor_path = 'preprocessor.pkl'
                if not os.path.exists(preprocessor_path):
                    raise FileNotFoundError(f"Archivo de preprocesador no encontrado en {preprocessor_path}")

                preprocessor = joblib.load(preprocessor_path)

                # Transformar los datos de entrada
                transformed_data = preprocessor.transform(input_data)

                # Generar predicciones
                probabilities = model.predict_proba(transformed_data)
                predictions = [
                    {f"Clase_{i + 1}": prob for i, prob in enumerate(prob_row)}
                    for prob_row in probabilities
                ]

                # Guardar las predicciones
                predictions_df = pd.DataFrame(predictions)
                output_path = os.path.join(output_folder, f"predictions_{file_name}")
                predictions_df.to_parquet(output_path)

                logging.info(f"Predicciones guardadas en: {output_path}")

                # Mover el archivo procesado a la carpeta de procesados
                processed_path = os.path.join(processed_folder, file_name)
                shutil.move(input_path, processed_path)
                logging.info(f"Archivo procesado movido a: {processed_path}")

            except Exception as e:
                logging.error(f"Error al procesar el archivo {file_name}: {e}")

if __name__ == "__main__":
    run_batch()
