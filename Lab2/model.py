import os
import joblib
import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_environment_variables():
    """
    Carga las variables de entorno necesarias para la ejecución.
    """
    load_dotenv()
    dataset_path = os.getenv("DATASET")
    target_column = os.getenv("TARGET")
    model_name = os.getenv("MODEL")
    trials = int(os.getenv("TRIALS", 10))
    return dataset_path, target_column, model_name, trials

def preprocess_data(df, target_column=None):
    """
    Preprocesa el conjunto de datos, aplicando transformaciones numéricas y categóricas.
    """
    X = df.drop(columns=[target_column]) if target_column else df
    y = df[target_column] if target_column else None

    # Identificación de columnas por tipo
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    # Pipelines para transformación
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Transformador por columnas
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Ajuste y transformación de datos
    X_processed = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, "preprocessor.pkl")
    logging.info("Preprocesador guardado en 'preprocessor.pkl'.")

    return X_processed, y

def train_and_optimize_model(X, y, model_name, trials=10):
    """
    Entrena y optimiza un modelo con RandomizedSearchCV.
    """
    # Diccionario de modelos disponibles
    models = {
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB()
    }

    # Hiperparámetros para búsqueda aleatoria
    param_grids = {
        "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]},
        "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5, 7]},
        "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
        "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
        "NaiveBayes": {}
    }

    # Selección del modelo
    model = models.get(model_name)
    if not model:
        raise ValueError(f"Modelo '{model_name}' no soportado.")

    # Búsqueda de hiperparámetros
    search = RandomizedSearchCV(model, param_grids[model_name], n_iter=trials, cv=3, verbose=1, random_state=42)
    search.fit(X, y)

    logging.info(f"Mejores hiperparámetros para {model_name}: {search.best_params_}")
    logging.info(f"Mejor puntuación para {model_name}: {search.best_score_:.4f}")

    return search.best_estimator_

def save_model(model, model_path):
    """
    Guarda el modelo entrenado en un archivo.
    """
    joblib.dump(model, model_path)
    logging.info(f"Modelo guardado en '{model_path}'.")

def main():
    """
    Función principal para ejecutar el entrenamiento del modelo.
    """
    dataset_path, target_column, model_name, trials = load_environment_variables()

    logging.info(f"Leyendo datos desde '{dataset_path}'.")
    df = pd.read_parquet(dataset_path)

    logging.info(f"Preprocesando datos con variable objetivo '{target_column}'.")
    X, y = preprocess_data(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"División de datos completada: {len(X_train)} entrenamiento, {len(X_test)} prueba.")

    logging.info(f"Iniciando entrenamiento y optimización del modelo '{model_name}'.")
    model = train_and_optimize_model(X_train, y_train, model_name, trials)

    model_path = "models/trained_model.joblib"
    save_model(model, model_path)

    logging.info("Evaluando modelo en conjunto de prueba.")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    logging.info(f"Precisión: {accuracy:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    logging.info(f"Matriz de confusión:\n{cm}")

if __name__ == "__main__":
    main()
