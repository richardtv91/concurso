import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Cargar el modelo
model_path = 'C:\\IA\\concurso\\mobilenetv2_covid_classifier.h5'  # Cambia la ruta si es necesario
model = load_model(model_path)

# Función para cargar y preprocesar imágenes
def load_and_preprocess_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen {path}")
    image = cv2.resize(image, (150, 150))  # Asegúrate de que el tamaño coincida con el tamaño de entrada del modelo
    image = np.stack([image]*3, axis=-1)  # Convertir a 3 canales
    image = image.astype(np.float32) / 255.0  # Normalizar
    return image

# Evaluar el modelo en un conjunto de prueba
def evaluate_model(test_data_dir):
    class_map = {
        'covid': 0,
        'viral_pneumonia': 1,
        'normal': 2
    }
    correct_predictions = 0
    total_images = 0

    for class_name, label in class_map.items():
        class_dir = os.path.join(test_data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Advertencia: directorio {class_dir} no existe, omitiendo clase.")
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_images += 1
                img_path = os.path.join(class_dir, filename)
                img = load_and_preprocess_image(img_path)
                img = np.expand_dims(img, axis=0)  # Añadir dimensión de batch
                prediction = model.predict(img)
                predicted_class = np.argmax(prediction, axis=1)[0]
                if predicted_class == label:
                    correct_predictions += 1

    accuracy = correct_predictions / total_images if total_images > 0 else 0
    print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Llama a la función de evaluación con la ruta de tu conjunto de prueba
evaluate_model('C:\\IA\\concurso\\test_data')  # Cambia la ruta a tu conjunto de prueba
