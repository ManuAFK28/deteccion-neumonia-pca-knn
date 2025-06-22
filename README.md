#Detección Automatizada de Neumonía en Radiografías de Tórax mediante PCA y KNN

Este proyecto implementa un sistema de detección automatizada de neumonía en radiografías de tórax utilizando técnicas de álgebra lineal computacional como el **Análisis de Componentes Principales (PCA)** y el algoritmo de clasificación **K-Vecinos Más Cercanos (KNN)**.

El sistema permite reducir la dimensionalidad de las imágenes médicas y entrenar un modelo que puede clasificar automáticamente nuevos casos con base en patrones visuales aprendidos.

---

## 📂 Estructura del proyecto

proyecto/
    detector.py # Código principal
    dataset/ # Carpeta contenedora del dataset
       imagenes/
           NORMAL/ # Radiografías de pacientes sanos (100 imágenes)
           PNEUMONIA/ # Radiografías con neumonía (100 imágenes)
     README.md # Este archivo

---

## Requisitos del sistema

### Requisitos de software

| Biblioteca         | Función principal                                              |
|--------------------|---------------------------------------------------------------|
| `numpy`            | Manipulación de matrices e imágenes vectorizadas              |
| `Pillow (PIL)`     | Carga, conversión a escala de grises y redimensionamiento     |
| `matplotlib.pyplot`| Visualización de imágenes y proyecciones PCA                  |
| `scikit-learn`     | Reducción de dimensionalidad (PCA), KNN, evaluación del modelo|

Puedes instalar los requerimientos con:

```bash
      pip install numpy pillow matplotlib scikit-learn
```
# Dataset utilizado
Este proyecto utiliza el conjunto de datos público Chest X-Ray Images (Pneumonia), disponible en Kaggle.

Link del dataset:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Importante:
Debes descargar manualmente el dataset y ubicarlo de la siguiente forma:

proyecto/
└── dataset/
    └── imagenes/
        ├── NORMAL/
        └── PNEUMONIA/

Para este proyecto se utilizan 100 imágenes normales y 100 con neumonía seleccionadas aleatoriamente desde la carpeta train/.

# Cómo ejecutar el programa

1. Clona o descarga este repositorio.
2. Instala las dependencias.
3. Coloca las imágenes en la ruta dataset/imagenes/ como se indicó.
4. Ejecuta el programa

El sistema entrenará el modelo, aplicará reducción de dimensionalidad con PCA y mostrará los resultados de clasificación con una matriz de confusión y métricas como precisión, recall y F1-score.

# Resultados esperados

- Visualización de las radiografías proyectadas en un plano 2D (PCA).
- Precisión promedio de clasificación: ~88% (según el conjunto de prueba).
- Evaluación automática del modelo con métricas confiables.

# Créditos
Este proyecto fue desarrollado como parte del curso Álgebra Lineal Computacional en la Universidad San Ignacio de Loyola, con fines académicos.
