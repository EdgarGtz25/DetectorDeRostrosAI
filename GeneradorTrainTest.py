import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Leer datos desde un archivo CSV
def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values  # Todas las columnas excepto la última
    y = df.iloc[:, -1].values   # Última columna
    return X, y

# Ejemplo de uso del perceptrón leyendo archivos CSV de entrenamiento y prueba
if __name__ == "__main__":
    # Proporciona la ruta correcta a tus archivos CSV
    train_filename = 'distancias_interpoladas.csv'
    
    # Cargar datos
    X, y = load_data(train_filename)
    print("Datos cargados:")
    print(X[:5])  # Imprimir las primeras 5 filas de características
    print(y[:5])  # Imprimir las primeras 5 etiquetas
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\nConjunto de entrenamiento y prueba divididos:")
    print("Tamaño del conjunto de entrenamiento:", X_train.shape)
    print("Tamaño del conjunto de prueba:", X_test.shape)
    
    # Escalar características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("\nCaracterísticas escaladas:")
    print(X_train[:5])  # Imprimir las primeras 5 filas de características escaladas
    
    # Inicializar y entrenar el Perceptrón
    perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    perceptron.fit(X_train, y_train)
    print("\nPerceptrón entrenado")
    print("Coeficientes:", perceptron.coef_)
    print("Intersección:", perceptron.intercept_)
    
    # Predecir en los datos de prueba
    y_pred = perceptron.predict(X_test)
    print("\nPredicciones en el conjunto de prueba:")
    print(y_pred[:5])  # Imprimir las primeras 5 predicciones
    
    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("\nEvaluación del modelo:")
    print("Precisión en los datos de prueba:", accuracy)
    print("Etiquetas verdaderas vs Predicciones:")
    print(list(zip(y_test[:5], y_pred[:5])))  # Imprimir las primeras 5 etiquetas verdaderas vs predicciones
