# ===============================================================
#  CLASIFICACIÓN BINARIA: LATE BLIGHT VS HEALTHY
#  Dataset: Kaggle con carpetas Train / Test / Valid
#  Modelo: MobileNetV2 + Transfer Learning
# ===============================================================

import kagglehub
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models


# ---------------------------------------------------------------
# 1. Descargar dataset
# ---------------------------------------------------------------
path = kagglehub.dataset_download("muhammadardiputra/potato-leaf-disease-dataset")
print("Dataset en:", path)

# IMPORTANTE: el dataset tiene otra carpeta llamada "Potato"
base_dir = os.path.join(path, "Potato")

train_dir = os.path.join(base_dir, "Train")
val_dir   = os.path.join(base_dir, "Valid")
test_dir  = os.path.join(base_dir, "Test")

print("Train:", train_dir)
print("Valid:", val_dir)
print("Test:", test_dir)

# Usaremos únicamente Healthy y Late Blight
CLASSES = ["Potato___healthy", "Potato___Late_blight"]


# ---------------------------------------------------------------
# 2. Generadores (solo 2 clases)
# ---------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_gen = ImageDataGenerator(rescale=1/255, rotation_range=20, horizontal_flip=True)
test_gen  = ImageDataGenerator(rescale=1/255)
val_gen   = ImageDataGenerator(rescale=1/255)

train_data = train_gen.flow_from_directory(
    train_dir,
    classes=CLASSES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    classes=CLASSES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    classes=CLASSES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)


# ---------------------------------------------------------------
# 3. MobileNetV2 preentrenado (no se entrena base)
# ---------------------------------------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Congelamos capas base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ---------------------------------------------------------------
# 4. Entrenar modelo
# ---------------------------------------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)


# ---------------------------------------------------------------
# 5. Evaluación final
# ---------------------------------------------------------------
loss, acc = model.evaluate(test_data)
print(f"\nPrecisión final del modelo: {acc:.4f}")


# ---------------------------------------------------------------
# 6. Guardar modelo
"""
model.save("modelo_papa_lateblight.h5")
print("Modelo guardado como modelo_papa_lateblight.h5")
"""
# ---------------------------------------------------------------
