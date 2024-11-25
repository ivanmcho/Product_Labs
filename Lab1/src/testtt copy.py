import pandas as pd
import optuna
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

# Parámetros del pipeline
target_column = 'Churn'  # Reemplaza con el nombre de tu columna objetivo
test_size = 0.2  # Proporción del conjunto de prueba

# Carga y Exploración de Datos
# Cargar los datos desde un archivo .parquet
df = pd.read_parquet("df_mflow_test.parquet")
print("Primeros registros del conjunto de datos:")
print(df.head())

# Chequear valores faltantes
print("Valores faltantes por columna:")
print(df.isnull().sum())

# Separar características (X) y la variable objetivo (y)
X = df.drop(target_column, axis=1)
y = df[target_column]

# Identificar automáticamente las columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns
print("Características numéricas:", numeric_features)
print("Características categóricas:", categorical_features)

# Preprocesamiento de Datos
# Definición de transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar transformers en un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Función de optimización de hiperparámetros usando Optuna
def objective(trial):
    # Definir los hiperparámetros a optimizar para RandomForest
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 10, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    # Modelo: RandomForest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    # Pipeline de entrenamiento
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    # Entrenar el modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predicción y evaluación
    y_pred = clf.predict(X_test)
    accuracy = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['accuracy']
    
    return accuracy

# Crear un estudio de Optuna
study = optuna.create_study(direction='maximize')

# Optimizar el modelo con Optuna
study.optimize(objective, n_trials=10)  # Número de pruebas que deseas realizar

# Mostrar los mejores hiperparámetros encontrados
print(f"Mejores hiperparámetros: {study.best_params}")
print(f"Mejor precisión en validación: {study.best_value}")

# Entrenamiento final con los mejores hiperparámetros encontrados
best_params = study.best_params
best_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    random_state=42
)

# Entrenar el modelo con los mejores hiperparámetros
clf_final = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', best_model)])

# Entrenar en el conjunto de datos completo
clf_final.fit(X_train, y_train)

# Guardar el modelo final
joblib.dump(clf_final, "RandomForest_Optuna_model.joblib")
print("Modelo final guardado.")

# Evaluación del modelo final
y_pred_final = clf_final.predict(X_test)
print("Resultados para el modelo optimizado:")
print(classification_report(y_test, y_pred_final))

# Exportar resultados a CSV o Markdown
results = {
    "Modelo": ["RandomForest (Optuna)"],
    "Precisión": [classification_report(y_test, y_pred_final, output_dict=True)["weighted avg"]["precision"]],
    "Recall": [classification_report(y_test, y_pred_final, output_dict=True)["weighted avg"]["recall"]],
    "F1-Score": [classification_report(y_test, y_pred_final, output_dict=True)["weighted avg"]["f1-score"]]
}

results_df = pd.DataFrame(results)
results_df.to_csv("resultados_modelo_Optuna.csv", index=False)
print("Resultados exportados a resultados_modelo_Optuna.csv")
