import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Leer datos desde un archivo CSV
def load_data(filename):
    print(f"Cargando datos desde {filename}...")
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values  # Todas las columnas excepto la última
    y = df.iloc[:, -1].values   # Última columna
    print("Datos cargados con éxito.")
    return X, y

# Ejemplo de uso del KNN leyendo archivos CSV de entrenamiento y prueba
if __name__ == "__main__":
    # Proporciona la ruta correcta a tus archivos CSV
    train_filename = 'distancias_interpoladas.csv'
    
    # Cargar datos
    X, y = load_data(train_filename)
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    print("Dividiendo los datos en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Datos divididos con éxito.")
    
    # Escalar características
    print("Escalando características...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Características escaladas con éxito.")
    
    # Inicializar y entrenar el KNN
    print("Inicializando y entrenando el clasificador KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)  # Puedes ajustar el número de vecinos (k)
    knn.fit(X_train, y_train)
    print("Clasificador KNN entrenado con éxito.")
    
    # Predecir en los datos de prueba
    print("Realizando predicciones en el conjunto de prueba...")
    y_pred = knn.predict(X_test)
    print("Predicciones realizadas con éxito.")
    
    # Evaluar el modelo
    print("Evaluando el modelo...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión en los datos de prueba: {accuracy}")
