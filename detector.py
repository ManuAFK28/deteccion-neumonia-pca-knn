import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Rutas de carpetas
carpeta_normal = 'imagenes/NORMAL'
carpeta_pneumonia = 'imagenes/PNEUMONIA'

# Parámetros
tamano = (64, 64)  # redimensionar imágenes
X = []
y = []

# Función para cargar imágenes
def cargar_imagenes(ruta, etiqueta):
    for archivo in os.listdir(ruta):
        if archivo.endswith('.jpeg') or archivo.endswith('.jpg') or archivo.endswith('.png'):
            img = Image.open(os.path.join(ruta, archivo)).convert('L')
            img = img.resize(tamano)
            X.append(np.array(img).flatten())
            y.append(etiqueta)

# Cargar normales = 0, neumonía = 1
cargar_imagenes(carpeta_normal, 0)
cargar_imagenes(carpeta_pneumonia, 1)

X = np.array(X)
y = np.array(y)

print(f"Matriz de imágenes: {X.shape}")
print(f"Etiquetas: {y.shape}")

# Aplicar PCA: reducir a 2 dimensiones para visualizar
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Varianza explicada por PCA:", pca.explained_variance_ratio_)

plt.figure(figsize=(8,6))
for clase, color, etiqueta in zip([0,1], ['blue', 'red'], ['Normal', 'Neumonía']):
    plt.scatter(X_pca[y == clase, 0], X_pca[y == clase, 1], c=color, label=etiqueta, alpha=0.6)

plt.title('Proyección PCA de radiografías de tórax')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True)
plt.show()

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Clasificador KNN
modelo_knn = KNeighborsClassifier(n_neighbors=3)
modelo_knn.fit(X_train, y_train)

# Evaluar
y_pred = modelo_knn.predict(X_test)
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

