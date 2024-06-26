from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import cv2
from joblib import load
import os
from scipy.interpolate import interp1d

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Crear el directorio de carga si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Cargar el modelo KNN y el escalador
model_filename = 'modelo_knn.joblib'
scaler_filename = 'scaler.joblib'

knn = load(model_filename)
scaler = load(scaler_filename)
print("Modelo KNN y escalador cargados exitosamente.")

def preprocess_image(image_path):
    # Leer la imagen en escala de grises
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro de desenfoque
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Utilizar el algoritmo de Canny para detectar los bordes
    edges = cv2.Canny(blurred, 50, 150)
    
    # Encontrar los contornos a partir de los bordes detectados
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Obtener las dimensiones de la imagen y calcular el centro
    height, width = gray.shape
    center_x, center_y = width // 2, height // 2
    
    # Lista para almacenar todas las distancias
    distances = []
    
    # Calcular la distancia desde cada punto de cada contorno hasta el centro de la imagen
    for contour in contours:
        for point in contour:
            x, y = point[0]
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)  # Distancia Euclidiana
            distances.append(distance)
    
    # Estandarizar la longitud de la serie de distancias a 2500
    if len(distances) > 0:
        f = interp1d(range(len(distances)), distances, kind='linear')
        standardized_distances = f(np.linspace(0, len(distances) - 1, 2500))
    else:
        standardized_distances = np.zeros(2500)  # Si no hay contornos, usar una serie de ceros
    
    # Escalar las características
    standardized_distances = scaler.transform([standardized_distances])
    
    return standardized_distances

@app.route('/predict', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    confidence = None
    if request.method == 'POST':
        # Guardar la imagen subida
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Preprocesar la imagen y hacer la predicción
            image = preprocess_image(image_path)
            print(f"Imagen preprocesada: {image}")
            prediction = knn.predict(image)[0]
            confidence = knn.predict_proba(image)[0].max() * 100  # Obtener el valor máximo de probabilidad y convertirlo a porcentaje
            print(f"Predicción: {prediction}, Confianza: {confidence:.2f}%")

    return render_template('index.html', prediction=prediction, image_path=image_path, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
