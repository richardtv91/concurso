import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

# Configuración
IMG_SIZE = 150
CLASS_NAMES = ['COVID-19', 'Neumonía viral', 'Normal']

# Función para cargar y preprocesar imágenes
def load_and_preprocess_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen {path}")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = np.stack([image]*3, axis=-1)  # Convertir a 3 canales
    image = image.astype(np.float32) / 255.0  # Normalizar a [0,1]
    return image

# Función para predecir la clase de una imagen
def predict_image(image_path):
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Añadir dimensión batch
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Función para abrir el diálogo de selección de archivos
def select_image():
    image_path = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        predicted_class_idx, confidence = predict_image(image_path)
        display_image(image_path, CLASS_NAMES[predicted_class_idx], confidence)

# Función para mostrar la imagen y la predicción
def display_image(image_path, predicted_class, confidence):
    # Cargar la imagen original
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.LANCZOS)  # Redimensionar para mostrar
    img_tk = ImageTk.PhotoImage(img)

    # Mostrar la imagen en la etiqueta
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Mantener una referencia a la imagen

    # Mostrar la predicción
    prediction_label.config(text=f"Predicción: {predicted_class} (Confianza: {confidence:.2f})")

# Cargar el modelo
model = load_model('mobilenetv2_covid_classifier.h5')  # Ajusta la ruta si es necesario

# Crear la ventana principal
root = Tk()
root.title("Clasificador de Imágenes COVID-19")

# Crear un botón para seleccionar la imagen
select_button = Button(root, text="Seleccionar Imagen", command=select_image)
select_button.pack(pady=10)

# Crear una etiqueta para mostrar la imagen
image_label = Label(root)
image_label.pack(pady=10)

# Crear una etiqueta para mostrar la predicción
prediction_label = Label(root, text="")
prediction_label.pack(pady=10)

# Iniciar el bucle principal de la interfaz gráfica
root.mainloop()
