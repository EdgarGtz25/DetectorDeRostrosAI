from pyspark.sql import SparkSession
import cv2
import os
import matplotlib.pyplot as plt
import logging

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

# 1. Configuración del Entorno Spark
spark = SparkSession.builder \
    .appName("Image Processing") \
    .getOrCreate()

# 2. Lectura de las Imágenes
image_paths = [
    "fotosAINAncy/ab67616d0000b273d1cbd51f69cb9803ea603c66.jpeg",
    # Añade más rutas de imágenes aquí
]
rdd = spark.sparkContext.parallelize(image_paths)

# 3. Definir el Directorio de Salida para las Imágenes Procesadas
output_dir = "D:/Edgar Gutierrez/Documents/Proyecti/fotosAINAncy"

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 4. Definir la Función de Procesamiento de Imágenes
def process_image(image_path):
    logging.debug(f"Processing image: {image_path}")
    
    # Leer la imagen desde la ruta
    image = cv2.imread(image_path)
    
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return None
    
    logging.debug("Image loaded successfully")
    
    # Lista para almacenar las imágenes intermedias
    processed_images = []
    step_titles = []

    # Paso 1: Imagen Original
    processed_images.append(image)
    step_titles.append("Original")

    # Paso 2: Aplicar Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    processed_images.append(blurred_image)
    step_titles.append("Gaussian Blur")

    # Paso 3: Convertir a Escala de Grises
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    processed_images.append(gray_image)
    step_titles.append("Grayscale")

    # Paso 4: Detección de Bordes con Canny
    edges_image = cv2.Canny(gray_image, 100, 200)
    processed_images.append(edges_image)
    step_titles.append("Canny Edges")

    # Guardar y mostrar las imágenes de cada paso
    for i, img in enumerate(processed_images):
        # Generar una nueva ruta para guardar la imagen procesada
        image_name = os.path.basename(image_path)
        step_name = step_titles[i].replace(" ", "_").lower()
        output_path = os.path.join(output_dir, f"{step_name}_{image_name}")
        
        # Guardar la imagen procesada
        cv2.imwrite(output_path, img)
        
        # Mostrar la imagen usando matplotlib
        plt.figure(figsize=(8, 6))
        if len(img.shape) == 2:  # Imagen en escala de grises
            plt.imshow(img, cmap='gray')
        else:  # Imagen en color
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        plt.title(step_titles[i])
        plt.axis('off')  # Ocultar los ejes
        plt.show()
    
    return output_path

# 5. Aplicar la Función de Procesamiento de Imágenes de Manera Distribuida
processed_images_rdd = rdd.map(process_image)

# Colectar las rutas de las imágenes procesadas
processed_images = processed_images_rdd.collect()

# Imprimir las rutas de las imágenes procesadas
logging.info("Imágenes procesadas y guardadas en las siguientes rutas:")
for path in processed_images:
    if path is not None:
        logging.info(path)

# Detener la sesión de Spark
spark.stop()
