# Modelo de Predicción de Precios de Viviendas

Este repositorio contiene un modelo de aprendizaje automático (ML) diseñado para predecir precios de casas, lo cual puede ser de gran ayuda en decisiones de inversión inmobiliaria con mayor precisión. El modelo utiliza técnicas de aprendizaje supervisado y el error cuadrático medio como métrica de evaluación. Además, se ha creado una API para facilitar el acceso y el uso del modelo.

## Datos
Los datos utilizados en este proyecto son de un repositorio público y están disponibles para uso libre. El dataset contiene información geográfica (latitud y longitud) junto con otras 8 características numéricas y una característica categórica ("ocean_proximity").

El tamaño del dataset es de aproximadamente 1.6 MB, por lo que se puede almacenar fácilmente de forma local para su uso.

## Exploración de Datos
Antes de entrenar el modelo, se exploraron y visualizaron los datos para obtener una mejor comprensión de las relaciones entre las variables. Se analizaron las correlaciones entre las diferentes características, así como la distribución de las mismas.

## Preparación de los Datos
Los datos se prepararon mediante la siguiente secuencia de pasos:

- Tratamiento de valores faltantes: Se utilizó un imputador (SimpleImputer) para rellenar los valores faltantes con la mediana de cada característica numérica.
- One-Hot Encoding: Se convirtió la característica categórica ("ocean_proximity") en variables binarias utilizando el método de One-Hot Encoding.
- Creación de características adicionales: Se agregaron tres características adicionales basadas en la relación entre otras características numéricas.
- Normalización de los datos: Se aplicó una estandarización para escalar las características numéricas.

## Modelado y Evaluación

Se probaron varios modelos para seleccionar el que mejor se ajusta a los datos. Se evaluaron los siguientes modelos:
- Regresión Lineal
- Árbol de Decisión
- Random Forest

El modelo de Random Forest fue seleccionado como el mejor modelo, ya que mostró un menor error cuadrático medio en la validación cruzada.

## Optimización de Hiperparámetros

Se realizó una optimización de hiperparámetros utilizando GridSearchCV para encontrar los mejores valores para los parámetros del modelo de Bosques Aleatorios.

## Evaluación Final

El modelo final se evaluó utilizando datos de prueba no vistos para obtener una estimación más realista del rendimiento. El error cuadrático medio obtenido en los datos de prueba fue de aproximadamente 48,230.85, lo que indica que el modelo tiene un buen rendimiento para predecir los precios de viviendas en nuevos datos.

## Uso de la API

El modelo entrenado y el preprocesamiento de datos se guardaron en archivos "my_model.pkl" y "my_pipeline.pkl", respectivamente. Estos archivos se pueden utilizar para cargar el modelo y el pipeline en la API y realizar predicciones en tiempo real.

Si tienes alguna pregunta o comentario, no dudes en contactarme.

- **Responsable del Proyecto:** María Marcela Balzarelli
- **Correo Electrónico:** marcebalzarelli@gmail.com
- **Linkedin:** https://www.linkedin.com/in/marcela-balzarelli/
