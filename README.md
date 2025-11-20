# Clasificador de Hojas de Papa – TensorFlow + CNN

Proyecto de clasificación de imágenes usando una red neuronal convolucional (CNN) entrenada con un dataset de Kaggle.  
El modelo se entrenó, evaluó y luego se utilizó para predecir imágenes nuevas.

---

## 🚀 Tecnologías

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-Learn  

---

## 🧠 Descripción del modelo

Se implementó una CNN con múltiples capas convolucionales, capas de pooling y capas densas al final.  
El modelo se entrenó con un dataset obtenido de Kaggle y se evaluó utilizando métricas como *accuracy* y *loss*.  
Finalmente, se incluyó un módulo para predecir imágenes individuales cargadas desde archivo.

---



## ▶️ Cómo correr el proyecto

### 1️⃣ Clonar el repositorio
```bash
git clone <url-del-repo>
cd <carpeta-del-repo>
````
2️⃣ Instalar dependencias
````bash
Copiar código
pip install -r requirements.txt
````

3️⃣ Entrenar el modelo
````bash
Copiar código
python src/train.py
````

4️⃣ Hacer predicciones
````bash
Copiar código
python app_predict/predict.py --image ruta/a/imagen.jpg
````

📥 Dataset
El dataset utilizado fue tomado de Kaggle:

Potato Leaf Disease Dataset – Muhammad Ardi Putra
https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset
