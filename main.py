# -*- coding: utf-8 -*-
"""Untitled2.ipynb


"""

#!pip install fastapi
#!pip install uvicorn

from fastapi import FastAPI
#import uvicorn
import joblib
import pandas as pd
import gdown

model_url = "https://drive.google.com/uc?id=1OnBifuBre4LLPKKXr0J2Jk9UcD6gpHYS"
pipeline_url = "https://drive.google.com/uc?id=1o5TEBKHDrV3qflGCcD-tRiJMrFxtL09y"

# Descargo los archivos desde Google Drive
gdown.download(model_url, "my_model.pkl", quiet=False)
gdown.download(pipeline_url, "my_pipeline.pkl", quiet=False)

# Cargo el modelo y el pipeline
model = joblib.load("my_model.pkl")
pipeline = joblib.load("my_pipeline.pkl")

app = FastAPI()

@app.post("/predict/")
async def predict(data: dict):
    """
    Endpoint para realizar predicciones.

    Recibe un JSON con los datos de entrada y devuelve la predicción del modelo.
    """
    try:
        # Convertir el JSON a un DataFrame de Pandas
        df = pd.DataFrame(data)

        # Aplicar el mismo pipeline de preprocesamiento utilizado durante el entrenamiento
        X_test_prepared = pipeline.transform(df)

        # Realizar la predicción con el modelo cargado
        predictions = model.predict(X_test_prepared)

        # Devolver las predicciones como JSON
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}

#uvicorn main:app --host 0.0.0.0 --port 8000

#import requests

# Ejemplo de datos para predecir
#data = [
  #  {"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41, "total_rooms": 880, "total_bedrooms": 129, "population": 322, "households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"},
  #  {"longitude": -122.22, "latitude": 37.86, "housing_median_age": 21, "total_rooms": 7099, "total_bedrooms": 1106, "population": 2401, "households": 1138, "median_income": 8.3014, "ocean_proximity": "NEAR BAY"},
   # {"longitude": -122.24, "latitude": 37.85, "housing_median_age": 52, "total_rooms": 1467, "total_bedrooms": 190, "population": 496, "households": 177, "median_income": 7.2574, "ocean_proximity": "NEAR BAY"}
#]

# Hacer la solicitud POST a la API para obtener las predicciones
#response = requests.post("http://localhost:8000/predict/", json=data)

# Mostrar las predicciones
#print(response.json())
