import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ---------------------------------------------------------
# 1. Cargar modelo entrenado
# ---------------------------------------------------------
model = tf.keras.models.load_model("../modelo_papa_lateblight.h5")

# Clases usadas en el entrenamiento
CLASSES = ["Potato___healthy", "Potato___Late_blight"]


# ---------------------------------------------------------
# 2. Función para predecir una imagen suelta
# ---------------------------------------------------------
def predecir_imagen(ruta_imagen):

    # Cargar imagen con el tamaño del modelo
    img = image.load_img(ruta_imagen, target_size=(224, 224))
    img_array = image.img_to_array(img)

    # Normalizar
    img_array = img_array / 255.0

    # Expandir dimensiones (Batch de 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    pred = model.predict(img_array)[0][0]

    # Interpetación
    if pred >= 0.5:
        clase = CLASSES[1]     # Late blight
        confianza = pred
    else:
        clase = CLASSES[0]     # Healthy
        confianza = 1 - pred

    print(f"\nPredicción: {clase}")
    print(f"Confianza: {confianza * 100:.2f}%\n")

    return clase, confianza


# ---------------------------------------------------------
# 3. Prueba local
# ---------------------------------------------------------
if __name__ == "__main__":
    ruta = "imagen_prueba.jpg"  # Cambia esto por tu imagen
    predecir_imagen(ruta)
