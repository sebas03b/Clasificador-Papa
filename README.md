Clasificador de hojas de papa – TensorFlow + CNN

Proyecto de clasificación de imágenes usando una red neuronal convolucional (CNN) entrenada con un dataset de Kaggle. El modelo se entrenó, evaluó y luego se utilizó para predecir imágenes nuevas.

🚀 Tecnologías

Python 

TensorFlow / Keras

NumPy

Matplotlib

Scikit-Learn

🧠 Descripción del modelo

Se implementó una CNN con múltiples capas convolucionales, de pooling y capas densas al final. El modelo se entrenó con un dataset obtenido de Kaggle y se evaluó utilizando métricas como accuracy y loss.
Finalmente, se agregó código para predecir imágenes individuales cargadas desde archivo.

📂 Estructura del proyecto

📁 proyecto/
  ├── app.py
  ├── modelo_papa_lateblight.h5
  ├── requirements.txt
  ├── app_predict/
  │   ├── imagen_prueba.jpg
  │   └── predict.py
  

▶️ Cómo correr el proyecto

Clonar el repositorio:

git clone <tu-repo>
cd <tu-repo>


Instalar dependencias:

pip install -r requirements.txt


Entrenar el modelo:

python src/train.py


Hacer predicciones:

python src/predict.py --image ruta/a/imagen.jpg

📥 Dataset

El dataset utilizado fue tomado de Kaggle:

Potato Leaf Disease Dataset by Muhammad Ardi Putra – Kaggle
(https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset)
