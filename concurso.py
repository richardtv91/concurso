import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
import dask.dataframe as dd
from dask.delayed import delayed
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import time

# --- 1. Configuración general y rutas ---
DATA_DIR = 'dataset_covid19'
IMG_SIZE = 150
BATCH_SIZE = 8  # Adaptado a RAM y CPU para eficiencia
NUM_CLASSES = 3
EPOCHS = 10  # Ajustable para pruebas y mejora

# Url referencia para dataset alternativo (en caso necesario)
COVID19_RADIOGRAPHY_DB_URL = 'https://www.kaggle.com/tawsifurrahman/covid19-radiography-database'
# Nota: Descarga manual o método alternativo para obtener datos porque Kaggle requiere login
# Aquí se asume que el usuario tiene los datos localmente organizados en DATA_DIR con carpetas: COVID, Normal, ViralPneumonia

# --- 2. Funciones de carga y preprocesamiento ---

def load_image_paths(data_dir):
    """
    Carga las rutas de las imágenes clasificadas en subcarpetas.
    Espera subcarpetas: COVID, Normal, ViralPneumonia
    Retorna DataFrame con columnas: filepath, label
    """
    labels_map = {'COVID':0, 'Normal':1, 'ViralPneumonia':2}
    data = []
    for label_name, label_id in labels_map.items():
        folder = os.path.join(data_dir, label_name)
        for img_path in glob(os.path.join(folder, '*.png')) + glob(os.path.join(folder, '*.jpg')):
            data.append({'filepath': img_path, 'label': label_id})
    df = pd.DataFrame(data)
    return df

def read_and_preprocess_image(path):
    """
    Lee imagen, redimensiona a IMG_SIZE x IMG_SIZE y la convierte a RGB.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Usamos grayscale por que radiografías son monocromo
    if img is None:
        raise FileNotFoundError(f'Imagen {path} no encontrada o inválida')
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img[..., np.newaxis]  # Añadir canal para tensor (H, W, 1)
    img = img / 255.0  # Normalización
    return img

def augment_image(image):
    """
    Aplica data augmentation ligera para evitar overfitting.
    """
    aug = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5)
    ])
    augmented = aug(image=image)
    return augmented['image']

# --- 3. Dataset personalizado con Dask (sin cargar todo en memoria) ---

def load_preprocess_dask(df):
    """
    Carga imágenes en paralelo usando dask.delayed
    Aplica preprocesamiento sin cargar todo en RAM a la vez
    """
    delayed_imgs = [delayed(read_and_preprocess_image)(path) for path in df['filepath']]
    imgs = dd.compute(*delayed_imgs)
    imgs = np.array(imgs)
    labels = df['label'].values
    return imgs, labels

# --- 4. Construcción del modelo MobileNetV2 sin top, entrada (150,150,1) adaptado para grayscale ---

def build_model():
    # MobileNetV2 preentrenado espera 3 canales, repetiremos canal para adaptar a 3 canales
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False  # Congelar base para transferencia aprendizaje

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])  # Replicar canal gris a 3 canales
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 5. Entrenamiento del modelo ---

def train_model(model, X_train, y_train, X_val, y_val):
    # Convertir datasets a tf.data para batching y shuffling eficiente sin gran uso de RAM
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=2)
    return history

# --- 6. Evaluación ---

def evaluate_model(model, X_test, y_test):
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    preds = model.predict(test_ds).argmax(axis=1)
    print("Reporte de clasificación:")
    print(classification_report(y_test, preds, target_names=['COVID', 'Normal', 'Neumonia']))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, preds))

# --- 7. Función principal ---

def main():
    print("Cargando rutas de imágenes...")
    df = load_image_paths(DATA_DIR)
    print(f"Datos cargados: {len(df)} imágenes")

    # Dividir dataset en train/val/test (70/15/15)
    train_val, test = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, stratify=train_val['label'], random_state=42)
    # 0.1765 ≈ 15% del total, para val

    print("Preprocesando imágenes de entrenamiento...")
    X_train, y_train = load_preprocess_dask(train)
    print("Preprocesando imágenes de validación...")
    X_val, y_val = load_preprocess_dask(val)
    print("Preprocesando imágenes de prueba...")
    X_test, y_test = load_preprocess_dask(test)

    print("Construyendo modelo...")
    model = build_model()
    print(model.summary())

    print("Entrenando modelo...")
    train_model(model, X_train, y_train, X_val, y_val)

    print("Evaluando modelo...")
    evaluate_model(model, X_test, y_test)

    # Guardar modelo para futuro despliegue
    model.save("covid_classifier_mobilenetv2.h5")
    print("Modelo guardado como covid_classifier_mobilenetv2.h5")

    # Levantar demo Streamlit simple
    launch_streamlit()

# --- 8. Interfaz simple con Streamlit para demo ---

def predict_image(model, image_path):
    img = read_and_preprocess_image(image_path)
    img = np.repeat(img, 3, axis=2)  # gris a 3 canales
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    classes = ['COVID-19', 'Normal', 'Neumonía viral']
    pred_label = classes[np.argmax(pred)]
    pred_confidence = np.max(pred)
    return pred_label, pred_confidence

def launch_streamlit():
    st.title("Clasificador COVID-19 / Neumonía / Normal - Radiografías de tórax")

    st.markdown("""
    Suba una radiografía de tórax para clasificar:
    - COVID-19
    - Neumonía viral
    - Pulmones normales

    Modelo MobileNetV2 optimizado para CPU y hardware limitado.
    """)

    uploaded_file = st.file_uploader("Seleccione imagen", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        with open("temp_img.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image("temp_img.png", caption='Imagen cargada', use_column_width=True)
        model = tf.keras.models.load_model("covid_classifier_mobilenetv2.h5")
        label, confidence = predict_image(model, "temp_img.png")
        st.markdown(f"### Predicción: **{label}** con confianza {confidence:.2%}")
        os.remove("temp_img.png")

if __name__ == '__main__':
    main()

