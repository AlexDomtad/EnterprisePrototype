#!/usr/bin/env python
# coding: utf-8

# # PROYECTO FINAL

# ## Carga de los datos

# In[5]:


import cv2
import os
import numpy as np 
import keras
from tensorflow import keras
import matplotlib.pyplot as plt
import glob
import tensorflow as tf


# In[6]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
plt.rcParams["figure.figsize"] = (20,6)


# In[7]:


import os
from tensorflow.keras.utils import get_file


# In[8]:


# Descomprimimos el archivo en tmp para visualizar
# !tar -xzf /root/.keras/datasets/simpsons_train.tar.gz -C /tmp/simpsons


# In[16]:


# Esta variable contiene un mapeo de número de clase a personaje.
# Utilizamos sólo los 18 personajes del dataset que tienen más imágenes.
MAP_CHARACTERS = {
    0: 'Empresa1', 1: 'Empresa2'
}

# Vamos a standarizar todas las imágenes a tamaño 64x64
IMG_SIZE = 200


# In[17]:


def load_train_set(dirname, map_characters, verbose=True):
    """Esta función carga los datos de training en imágenes.
    
    Como las imágenes tienen tamaños distintas, utilizamos la librería opencv
    para hacer un resize y adaptarlas todas a tamaño IMG_SIZE x IMG_SIZE.
    
    Args:
        dirname: directorio completo del que leer los datos
        map_characters: variable de mapeo entre labels y personajes
        verbose: si es True, muestra información de las imágenes cargadas
     
    Returns:
        X, y: X es un array con todas las imágenes cargadas con tamaño
                IMG_SIZE x IMG_SIZE
              y es un array con las labels de correspondientes a cada imagen
    """
    X_train = []
    y_train = []
    for label, character in map_characters.items():        
        files = os.listdir(os.path.join(dirname, character))
        images = [file for file in files if file.endswith("jpg")]
        if verbose:
          print("Leyendo {} imágenes encontradas de {}".format(len(images), character))
        for image_name in images:
            image = cv2.imread(os.path.join(dirname, character, image_name))
            X_train.append(cv2.resize(image,(IMG_SIZE, IMG_SIZE)))
            y_train.append(label)
    return np.array(X_train), np.array(y_train)


# In[20]:


def load_test_set(dirname, map_characters, verbose=True):
    """Esta función carga los datos de prueba desde un directorio con imágenes sueltas."""
    X_test = []
    y_test = []
    reverse_dict = {v: k for k, v in map_characters.items()}
    for filename in glob.glob(dirname + '/*.*'):
        image_name = os.path.basename(filename)
        char_name = "_".join(image_name.split('_')[:-1])
        if char_name in reverse_dict:
            image = cv2.imread(filename)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            X_test.append(image)
            y_test.append(reverse_dict[char_name])
    if verbose:
        print("Leídas {} imágenes de test".format(len(X_test)))
    return np.array(X_test), np.array(y_test)


# In[21]:


# Cargamos los datos. 
# los de los ficheros donde hayas descargado los datos.
DATASET_TRAIN_PATH_COLAB = "C:/Users/alexd/.keras/datasets/datasets/TrainEnterprise"
DATASET_TEST_PATH_COLAB = "C:/Users/alexd/.keras/datasets/datasets/Enterprisetest"

X, y = load_train_set(DATASET_TRAIN_PATH_COLAB, MAP_CHARACTERS)
X_t, y_t = load_test_set(DATASET_TEST_PATH_COLAB, MAP_CHARACTERS)


# In[22]:


# Vamos a barajar aleatoriamente los datos. Esto es importante ya que si no
# lo hacemos y, por ejemplo, cogemos el 20% de los datos finales como validation
# set, estaremos utilizando solo un pequeño número de personajes, ya que
# las imágenes se leen secuencialmente personaje a personaje.
perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]


# In[23]:


plt.imshow(X_t[1])


# In[24]:


plt.imshow(np.flip(X_t[1], axis=-1) ) 


# ## Ejercicio
# 
# Utilizando Convolutional Neural Networks con Keras, entrenar un clasificador que sea capaz de reconocer personajes en imágenes de los Simpsons con una accuracy en el dataset de test de, al menos, **85%**. Redactar un informe analizando varias de las alternativas probadas y los resultados obtenidos.
# 
# A continuación se detallan una serie de aspectos para ser analizados en vuestro informe:
# 
# *   Análisis de los datos a utilizar.
# *   Análisis de resultados, obtención de métricas de *precision* y *recall* por clase y análisis de qué clases obtienen mejores o peores resultados.
# *   Análisis visual de los errores de la red. ¿Qué tipo de imágenes o qué personajes dan más problemas a nuestro modelo?
# *   Comparación de modelos CNNs con un modelo de Fully Connected para este problema.
# *   Utilización de distintas arquitecturas CNNs, comentando aspectos como su profundidad, hiperparámetros utilizados, optimizador, uso de técnicas de regularización, *batch normalization*, etc.
# *   Utilización de *data augmentation*. Esto puede conseguirse con la clase [ImageDataGenerator](https://keras.io/preprocessing/image/#imagedatagenerator-class) de Keras.
# 
# Notas: 
# * Recuerda partir los datos en training/validation para tener una buena estimación de los valores que nuestro modelo tendrá en los datos de test, así como comprobar que no estamos cayendo en overfitting. Una posible partición puede ser 80 / 20.
# * No es necesario mostrar en el notebook las trazas de entrenamiento de todos los modelos entrenados, si bien una buena idea seria guardar gráficas de esos entrenamientos para el análisis. Sin embargo, **se debe mostrar el entrenamiento completo del mejor modelo obtenido y la evaluación de los datos de test con este modelo**.
# * Las imágenes **no están normalizadas**. Hay que normalizarlas como hemos hecho en trabajos anteriores.
# * El test set del problema tiene imágenes un poco más "fáciles", por lo que es posible encontrarse con métricas en el test set bastante mejores que en el training set.

# ## Normalizamos nuestras imagenes en X

# In[25]:


# Normalizar imágenes
X_normalized = X / 255.0


# ## Analisis de datos
# #### Se hace un análisis de los datos ya que influirá mucho tener una cantidad balanceada de datos para poder entrenar nuestra IA

# In[26]:


import numpy as np
from collections import Counter



# Se obtienen la cantidad de datos por clase usando Counter
class_counts = Counter(y)

# Imprime la cantidad de datos por clase
for cls, count in class_counts.items():
    print(f"{MAP_CHARACTERS[cls]}: {count} imágenes")


# In[27]:


import matplotlib.pyplot as plt

# Datos
classes = [MAP_CHARACTERS[cls] for cls in class_counts.keys()]
counts = list(class_counts.values())

# Crear gráfico de barras
plt.figure(figsize=(10, 6))
bars = plt.bar(classes, counts, color='skyblue')
plt.xlabel('Personajes')
plt.ylabel('Número de Imágenes')
plt.title('Distribución de Imágenes por Clase')
plt.xticks(rotation=45, ha='right')  # Rotar etiquetas en el eje x para mayor legibilidad

# Añadir etiquetas con números
for bar, count in zip(bars, counts):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(count, 2), ha='center', va='bottom', color='black')

plt.tight_layout()

# Mostrar el gráfico
plt.show()


# ## Data augmentation 
# #### Se utiliza para llenar las carpetas que contienen muy pocas imagenes de muestra, con eso balancearemos nuestros datos de entrenamiento, al final del proceso monitoreamos como se encuentran nuestros datos distribuidos, si se encuentran bien, seguimos con el siguiente paso.
# 

# In[28]:


from keras_preprocessing.image import ImageDataGenerator

# Crear un generador de imágenes con data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Crear un array para almacenar las imágenes aumentadas y sus etiquetas correspondientes
augmented_images = []
augmented_labels = []

# Definir el número objetivo de ejemplos por clase
target_count = 400

# Iterar sobre cada clase
for label in np.unique(y):
    # Seleccionar imágenes de la clase actual
    class_images = X_normalized[y == label]
    
    # Calcular cuántas imágenes adicionales se necesitan para alcanzar el objetivo
    images_needed = target_count - len(class_images)
    
    # Aplicar data augmentation solo si se necesitan más imágenes
    if images_needed > 0:
        # Seleccionar imágenes aleatorias de la clase actual para aplicar data augmentation
        selected_images = class_images[np.random.choice(len(class_images), images_needed, replace=True)]
        
        # Iterar sobre cada imagen seleccionada y aplicar data augmentation
        for image in selected_images:
            image = image.reshape((1,) + image.shape)  # Añadir dimensión para que sea compatible con el generador
            for batch in datagen.flow(image, batch_size=1):
                augmented_images.append(batch[0])
                augmented_labels.append(label)
                break  # Necesario para evitar un bucle infinito debido a la generación continua

# Concatenar las imágenes originales con las aumentadas
X_augmented = np.concatenate([X_normalized, np.array(augmented_images)])
y_augmented = np.concatenate([y, np.array(augmented_labels)])

# Verificar la cantidad de ejemplos por clase después de la aplicación de data augmentation
class_counts = np.bincount(y_augmented)
print("Número de ejemplos por clase después de data augmentation:")
# Imprimir la cantidad de imágenes por clase después del aumento de datos
for label, count in enumerate(class_counts):
    print(f"{MAP_CHARACTERS[label]}: {count}")



# In[29]:


import matplotlib.pyplot as plt
from collections import Counter

# Obtener la cantidad de datos por clase después de data augmentation
class_counts_augmented = Counter(y_augmented)

# Datos para el gráfico
classes_augmented = [MAP_CHARACTERS[cls] for cls in class_counts_augmented.keys()]
counts_augmented = list(class_counts_augmented.values())

# Crear gráfico de barras
plt.figure(figsize=(10, 6))
bars = plt.bar(classes_augmented, counts_augmented, color='lightcoral')
plt.xlabel('Personajes')
plt.ylabel('Número de Imágenes')
plt.title('Distribución de Imágenes por Clase después de Data Augmentation')
plt.xticks(rotation=45, ha='right')  # Rotar etiquetas en el eje x para mayor legibilidad

# Añadir etiquetas con números
for bar, count in zip(bars, counts_augmented):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(count, 2), ha='center', va='bottom', color='black')

plt.tight_layout()

# Mostrar el gráfico
plt.show()


# ## Revisión de datos
# #### Imprimimos imagenes aleatorios para poder verificar la calidad de nuestras dataset de entrenamiento

# In[30]:


import random
import matplotlib.pyplot as plt

# Número de imágenes que se desean seleccionar
num_images_to_display = 5

# Obtener índices aleatorios para seleccionar imágenes
random_indices = random.sample(range(len(X_augmented)), num_images_to_display)

# Mostrar las imágenes seleccionadas con nombres de clase
for i, index in enumerate(random_indices, 1):
    plt.subplot(1, num_images_to_display, i)
    plt.imshow(X_augmented[index])
    plt.title(f" {MAP_CHARACTERS[y_augmented[index]]}")
    plt.axis('off')

plt.show()


# ## Separacion de sets (entrenamiento y validación)
# #### Dividiendo en una proporción 80%-20% para poder tener una distribucion optima, al final se validan la forma de cada set.

# In[31]:


from sklearn.model_selection import train_test_split

# Especifica la proporción deseada para el conjunto de validación (por ejemplo, 20%)
validation_split = 0.2

# Divide los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_augmented, y_augmented, test_size=validation_split, random_state=42)

# Imprime las formas de los conjuntos resultantes
print("Forma del conjunto de entrenamiento:", X_train.shape)
print("Forma del conjunto de validación:", X_val.shape)


# ## Arquitectura del modelo
# #### En este modelo quise experimentar con una arquitectura mucho más compleja, utilizando un 3 capas convolucionales y 2 FC, resultando en un gran resultado al probar con imagenes externas a nuestros sets de test, creando un earlystop para que no cometamos overfitting.

# In[32]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

# Tamaño de la imagen
IMG_SIZE = 200

# Utiliza una red preentrenada como base (ejemplo: ResNet50)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Congela las capas preentrenadas
for layer in base_model.layers:
    layer.trainable = False

# Define el modelo
model = Sequential()

# Añade la red preentrenada como base
model.add(base_model)

# Añade capas adicionales para capturar detalles finos
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Capa de aplanamiento
model.add(GlobalAveragePooling2D())

# Capa fully connected 1
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Capa fully connected 2
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Capa de salida
model.add(Dense(len(MAP_CHARACTERS), activation='softmax'))

# Compila el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Agrega EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Muestra la arquitectura del modelo
model.summary()


# ## Entrenamiento del modelo

# In[33]:


# Entrena el modelo
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stop])


# ## Validamos con nuestros set de test
# #### Obtenemos una salida muy pobre, pero en las siguientes pruebas tiene un rendimiento increible.

# In[34]:


# Normalizar el conjunto de prueba dividiendo por 255
X_t_normalized = X_t / 255.0

# Evaluar el modelo en el conjunto de prueba normalizado
test_loss, test_accuracy = model.evaluate(X_t_normalized, y_t)
print(f"Accuracy en conjunto de prueba: {test_accuracy * 100:.2f}%")


# In[35]:


from sklearn.metrics import confusion_matrix

# Realiza predicciones en el conjunto de prueba
y_test_pred = model.predict(X_t)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Calcula la matriz de confusión para el conjunto de prueba
conf_matrix_test = confusion_matrix(y_t, y_test_pred_classes)

# Imprime la matriz de confusión para el conjunto de prueba
print("Matriz de confusión en el conjunto de prueba:")
print(conf_matrix_test)


# ## Pruebas prácticas
# 
# #### Se hace una prueba con imagenes fuera de nuestros sets, logrando predecir 4/4 imágenes.

# In[37]:


import cv2
import numpy as np

# Función para cargar y preprocesar una imagen
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (200, 200))
    img = img / 255.0  # Normalizar la imagen
    img = np.expand_dims(img, axis=0)  # Añadir dimensión de lote
    return img

# Ruta de la imagen que deseas probar
#image_path = "C:\\Users\\alexd\\.keras\\datasets\\datasets\\symbol\\symbol_bad\\simboloMal.jpg"
image_path = "C:\\Users\\alexd\\.keras\\datasets\\datasets\\factura1.jpg"

# Preprocesar la imagen
image = preprocess_image(image_path)

# Realizar la predicción con el modelo original
predictions = model.predict(image)

# Obtener la clase predicha
predicted_class = np.argmax(predictions)

# Imprimir el resultado
print(f"Clase predicha: {MAP_CHARACTERS[predicted_class]}")

# Mostrar la imagen
plt.imshow(image[0])
plt.title("Imagen de entrada")
plt.show()


# ## Métricas de precisión y recall por clase

# In[38]:


from sklearn.metrics import classification_report, confusion_matrix

# Realiza predicciones en el conjunto de validación
y_pred = model.predict(X_val)

# Convierte las predicciones a clases
y_pred_classes = np.argmax(y_pred, axis=1)

# Imprime el informe de clasificación
print(classification_report(y_val, y_pred_classes, target_names=list(MAP_CHARACTERS.values())))

# Imprime la matriz de confusión
conf_matrix = confusion_matrix(y_val, y_pred_classes)
print("Matriz de confusión:")
print(conf_matrix)


# ## Análisis visual de errores

# In[41]:


# Realiza predicciones en el conjunto de validación
y_pred = model.predict(X_val)

# Convierte las predicciones a clases
y_pred_classes = np.argmax(y_pred, axis=1)

# Encuentra índices de imágenes mal clasificadas
misclassified_indices = np.where(y_pred_classes != y_val)[0]

# Muestra algunas imágenes mal clasificadas
num_images_to_display = 5

for i, index in enumerate(misclassified_indices[:num_images_to_display], 1):
    plt.subplot(1, num_images_to_display, i)
    plt.imshow(X_val[index])
    plt.title(f"True: {MAP_CHARACTERS[y_val[index]]}\nPredicted: {MAP_CHARACTERS[y_pred_classes[index]]}")
    plt.axis('off')

plt.show()

