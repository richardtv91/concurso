import os
import cv2
import numpy as np
import tensorflow as tf
import albumentations as A
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Forzar uso solo CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMG_SIZE = 150
BATCH_SIZE = 8
EPOCHS = 15
DATA_DIR = "dataset"  # Asegúrate que contiene subcarpetas covid, viral_pneumonia, normal
MODEL_SAVE_PATH = "mobilenetv2_covid_classifier.h5"
CLASS_NAMES = ['COVID-19', 'Neumonía viral', 'Normal']

augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(rotate=20, scale=(0.9, 1.1), p=0.5),  # Eliminado 'shift_limit'
])

def load_and_preprocess_image(path, augment=False):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen {path}")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = np.stack([image]*3, axis=-1)
    if augment:
        augmented = augmentation(image=image)
        image = augmented['image']
    image = image.astype(np.float32) / 255.0
    return image

def get_image_paths_and_labels(data_dir):
    paths = []
    labels = []
    class_map = {
        'covid': 0,
        'viral_pneumonia': 1,
        'normal': 2
    }
    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Advertencia: directorio {class_dir} no existe, omitiendo clase.")
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(class_dir, filename))
                labels.append(label)
    return paths, labels

def tf_data_generator(paths, labels, batch_size, augment=False):
    dataset_size = len(paths)
    def gen():
        while True:
            indices = list(range(dataset_size))
            random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                batch_indices = indices[start:start+batch_size]
                batch_images = []
                batch_labels = []
                for idx in batch_indices:
                    img = load_and_preprocess_image(paths[idx], augment=augment)
                    batch_images.append(img)
                    batch_labels.append(labels[idx])
                yield np.array(batch_images), np.array(batch_labels)
    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.int32),
        output_shapes=([None, IMG_SIZE, IMG_SIZE, 3], [None,])
    )

def build_model(num_classes=3):
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def main():
    paths, labels = get_image_paths_and_labels(DATA_DIR)
    if len(paths) == 0:
        print("Error: No se encontraron imágenes. Verifica estructura dataset/* y que las carpetas contengan imágenes.")
        return
    # Barajar los datos antes de partir
    combined = list(zip(paths, labels))
    random.shuffle(combined)
    paths[:], labels[:] = zip(*combined)

    total = len(paths)
    split = int(0.8 * total)
    train_paths, val_paths = paths[:split], paths[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    if len(val_paths) == 0:
        print("Error: conjunto de validación vacío. Reduce tamaño de entrenamiento.")
        return

    print(f"Total: {total}, Train: {len(train_paths)}, Valid: {len(val_paths)}")

    train_dataset = tf_data_generator(train_paths, train_labels, BATCH_SIZE, augment=True)
    val_dataset = tf_data_generator(val_paths, val_labels, BATCH_SIZE, augment=False)

    train_steps = max(len(train_paths) // BATCH_SIZE, 1)
    val_steps = max(len(val_paths) // BATCH_SIZE, 1)

    model = build_model(num_classes=3)
    model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Comentar el checkpoint para probar sin él
    # checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    # early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    # Entrenamiento sin callbacks
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        # callbacks=[checkpoint, early_stop],
        verbose=2
    )
    print(f"Modelo guardado en: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()

