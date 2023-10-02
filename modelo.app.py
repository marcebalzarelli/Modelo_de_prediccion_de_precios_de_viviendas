import streamlit as st
import joblib
import pandas as pd
from modelo_de_predicción_de_precio_viviendas import CombinedAttributesAdder
import folium
import plotly.express as px
import matplotlib.pyplot as plt
import geopandas as gpd
import zipfile
import requests
import os

# URLs de los archivos en GitHub
zip_model_url = "https://github.com/marcebalzarelli/Modelo_de_prediccion_de_precios_de_viviendas/raw/main/my_model.zip"
pkl_model_url = "https://github.com/marcebalzarelli/Modelo_de_prediccion_de_precios_de_viviendas/raw/main/my_pipeline.pkl"

@st.cache
def cargar_modelos():
    # Descargo y descomprimo el modelo ZIP
    zip_model_dir = "zip_model"  # Carpeta donde se extraerá el modelo ZIP
    os.makedirs(zip_model_dir, exist_ok=True)
    
    zip_model_path = os.path.join(zip_model_dir, "my_model.zip")
    response = requests.get(zip_model_url)
    with open(zip_model_path, "wb") as zip_file:
        zip_file.write(response.content)
    
    with zipfile.ZipFile(zip_model_path, "r") as zip_ref:
        zip_ref.extractall(zip_model_dir)
    
    pkl_model_path = "my_model.pkl"
    response = requests.get(pkl_model_url)
    with open(pkl_model_path, "wb") as pkl_file:
        pkl_file.write(response.content)
    
    # Cargo ambos modelos
    loaded_zip_model = joblib.load(os.path.join(zip_model_dir, "my_model.pkl"))
    loaded_pkl_model = joblib.load("my_model.pkl")
    
    return loaded_zip_model, loaded_pkl_model

zip_model, pkl_model = cargar_modelos()# Llamo a la función para cargar los modelos


def predict(data):# Creo la función para predecir
    try:
        X_test_prepared = pipeline.transform(data) # Aplico el mismo pipeline de preprocesamiento utilizado durante el entrenamiento

        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)# Creo una instancia de CombinedAttributesAdder

        X_test_prepared = attr_adder.transform(X_test_prepared)# Aplico la transformación de CombinedAttributesAdder

        predictions = model.predict(X_test_prepared)# Realizo la predicción con el modelo 

        return predictions[0]
    except ValueError as ve:
        return f"Error de valor: {ve}"
    except Exception as e:
        return f"Error desconocido: {str(e)}"

st.title("Predicción de Precios de Viviendas")# Configuro la interfaz

# Creo formulario para ingresar datos
st.sidebar.header("Ingrese los datos:")
longitud = st.number_input("Longitud", value=0.0)
latitud = st.number_input("Latitud", value=0.0)
edad_media_de_la_vivienda = st.number_input("Edad Media de la Vivienda", value=0.0)
total_de_habitaciones = st.number_input("Total de Habitaciones", value=0.0)
total_de_dormitorios = st.number_input("Total de Dormitorios", value=0.0)
poblacion = st.number_input("Población", value=0.0)
hogares = st.number_input("Hogares", value=0.0)
ingreso_mediano = st.number_input("Ingreso Mediano", value=0.0)
valor_mediano_de_la_casa = st.number_input("Valor Mediano de la Casa", value=0.0)
proximidad_al_oceano = st.selectbox("Proximidad al Océano", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY"])

if st.sidebar.button("Realizar Predicción"):
    # Creo diccionario con los datos ingresados
    data = {
        "longitud": [longitud],
        "latitud": [latitud],
        "edad_media_de_la_vivienda": [edad_media_de_la_vivienda],
        "total_de_habitaciones": [total_de_habitaciones],
        "total_de_dormitorios": [total_de_dormitorios],
        "poblacion": [poblacion],
        "hogares": [hogares],
        "ingreso_mediano": [ingreso_mediano],
        "valor_mediano_de_la_casa": [valor_mediano_de_la_casa],
        "proximidad_al_oceano": [proximidad_al_oceano]
    }

    datosingresados = pd.DataFrame(data) # Convierto el diccionario a un DataFrame de Pandas

    predictions = predict(datosingresados)# Realizo predicción

    st.write("Predicción de Precio de Vivienda:", predictions[0])


st.header("Visualizaciones Interactivas")# Título de las visualizaciones

# URL del archivo CSV en GitHub
url = "https://raw.githubusercontent.com/marcebalzarelli/Modelo_de_prediccion_de_precios_de_viviendas/main/housing.csv"

@st.cache
def cargar_datos(url):
    df = pd.read_csv(url)
    return df

df = cargar_datos(url)# Llamo a la función para cargar los datos desde la URL

# Mapa interactivo con Folium
st.header("Distribución de Viviendas")
m = folium.Map(location=[36.76, -119.72], zoom_start=10)
for i in range(len(df)):
    folium.Marker([df['latitude'][i], df['longitude'][i]], popup=f'Vivienda {i+1}').add_to(m)
st.write(m)

# Gráfico interactivo con Plotly
st.header("Relación entre Ingresos y Valor de Viviendas")
fig = px.scatter(df, x='median_income', y='median_house_value', color='ocean_proximity')
st.plotly_chart(fig)

# Gráfico con Matplotlib
st.header("Distribución del Valor de Viviendas")
fig, ax = plt.subplots(figsize=(10, 10))
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
gdf.plot(column='median_house_value', cmap='coolwarm', markersize=10, ax=ax, legend=True)
st.pyplot(fig)

# Gráfico con GeoPandas
st.header("Ubicación de Viviendas")
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(column='median_house_value', cmap='coolwarm', markersize=10, ax=ax, legend=True)
st.pyplot(fig)
