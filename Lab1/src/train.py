# src/train.py
import pandas as pd
import joblib
import optuna
import sys
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def train(input_file, model_file, params_file):
    # Cargar el dataset preprocesado desde el archivo CSV
    df = pd.read_csv(input_file)
    print("Columnas en el archivo de entrada:", df.columns.tolist())  # Validación de columnas

    # Leer los hiperparámetros y los parámetros de preprocessing
    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    # Validación de columnas antes de continuar
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Las siguientes columnas faltan en el archivo de entrada: {missing_features}")

    X = df[features]
    y = df[target]

    # Dividir en conjuntos de entrenamiento y validación
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=params['train']['test_size'], random_state=params['train']['random_state']
    # )

    X_train, X_test, y_train, y_test= train_test_split(
        X, y, test_size=params['train']['test_size'], random_state=params['train']['random_state']
    )

    # Definir la función de objetivo para Optuna
    def objective(trial):
        # Escoger el modelo para optimizar
        model_name = trial.suggest_categorical("model", ["LinearRegression", "RandomForest", "GradientBoosting"])

        if model_name == "LinearRegression":
            model = LinearRegression()
        elif model_name == "RandomForest":
            n_estimators = trial.suggest_int("n_estimators", 50, 200, step=50)
            max_depth = trial.suggest_int("max_depth", 10, 50, step=10)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0, n_jobs=-1)
        elif model_name == "GradientBoosting":
            learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.1)
            n_estimators = trial.suggest_int("n_estimators", 50, 200, step=50)
            model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, random_state=0)

        # Crear el pipeline con el escalado y el modelo
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # Entrenar el modelo
        pipeline.fit(X_train, y_train)

        # Evaluar el rendimiento en el conjunto de prueba
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    # Configurar el estudio de Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    # Mostrar los mejores hiperparámetros
    print("Mejores hiperparámetros:", study.best_params)
    print("Mejor MSE:", study.best_value)

    # Obtener el mejor modelo con los mejores hiperparámetros
    best_params = study.best_params
    if best_params['model'] == "LinearRegression":
        best_model = LinearRegression()
    elif best_params['model'] == "RandomForest":
        best_model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                        max_depth=best_params['max_depth'], random_state=0, n_jobs=-1)
    elif best_params['model'] == "GradientBoosting":
        best_model = GradientBoostingRegressor(learning_rate=best_params['learning_rate'],
                                            n_estimators=best_params['n_estimators'], random_state=0)

    # Crear y ajustar el pipeline con el mejor modelo
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', best_model)
    ])

    best_pipeline.fit(X_train, y_train)

    # Guardar el pipeline completo con el mejor modelo
    joblib.dump(best_pipeline, model_file)
    print("Modelo guardado como 'mejor_modelo_pipeline.pkl'")

    # Visualizar la convergencia de los hiperparámetros
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Curva de convergencia de los hiperparámetros")
    plt.show()

    # Visualizar la importancia de los hiperparámetros
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Importancia de los hiperparámetros")
    plt.show()

    # Evaluación del mejor modelo en el conjunto de prueba
    y_pred = best_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Error cuadrático medio (MSE) del mejor modelo:", mse)
    print("Coeficiente de determinación (R2) del mejor modelo:", r2)

    # # Guardar el pipeline completo con el mejor modelo
    # joblib.dump(best_model, model_file)
    # print("Modelo guardado como 'mejor_modelo_pipeline.pkl'")




if __name__ == "__main__":
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    params_file = sys.argv[3]

    train(input_file, model_file, params_file)
