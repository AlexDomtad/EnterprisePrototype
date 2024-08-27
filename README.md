# Enterprise Prototype

## Descripción del Proyecto

Este proyecto utiliza Redes Neuronales Convolucionales (CNNs) para entrenar un modelo capaz de clasificar imágenes de empresas. El modelo fue desarrollado utilizando Keras y TensorFlow y entrenado con un dataset personalizado de imágenes.

## Estructura del Proyecto

- **Carga de Datos**: Se cargan y procesan las imágenes, ajustando su tamaño a 200x200 píxeles.
- **Normalización**: Las imágenes se normalizan dividiendo los valores de los píxeles por 255.
- **Aumento de Datos (Data Augmentation)**: Se utilizan técnicas de aumento de datos para equilibrar las clases del dataset.
- **División del Dataset**: Se divide el conjunto de datos en sets de entrenamiento y validación en una proporción 80%-20%.
- **Arquitectura del Modelo**: Se utiliza una red CNN con capas convolucionales y fully connected, basada en la arquitectura ResNet50.
- **Entrenamiento**: El modelo se entrena utilizando EarlyStopping para evitar el overfitting.
- **Evaluación**: Se evalúa el modelo con el set de prueba y se muestran métricas como precisión, recall y matriz de confusión.

## Requisitos

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Scikit-learn

## Instalación

Clona este repositorio y asegúrate de tener instaladas las dependencias necesarias:

```bash
git clone https://github.com/tu-usuario/Enterprise-Prototype.git
cd Enterprise-Prototype
pip install -r requirements.txt
```

## Uso
El modelo entrenado es capaz de clasificar con alta precisión imágenes de empresas dentro del set de prueba. Se observó un rendimiento excelente con imágenes externas al dataset, logrando una predicción correcta en el 100% de los casos probados.

Autor
Alexis Dominguez
