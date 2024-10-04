#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

# Función para aplicar DBSCAN, filtrar los datos y extraer el fondo (background)
def apply_dbscan_filter(df, eps=0.2, min_samples=20):
    """
    Aplica DBSCAN a un DataFrame para identificar outliers, filtrar los datos y extraer el fondo.

    :param df: DataFrame con los datos a los que se les aplicará DBSCAN.
    :param eps: Parámetro de distancia máxima para el DBSCAN.
    :param min_samples: Mínimo número de muestras en un vecindario para que un punto no sea considerado outlier.
    :return: DataFrame filtrado (sin outliers), Background_df (fondo) con las diferencias.
    """
    # Extraer coordenadas
    Xs = df['x'].values
    Ys = df['y'].values

    # Crear la matriz de coordenadas
    coordinates = np.column_stack((Xs, Ys))

    # Aplicar DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)

    # Extraer las etiquetas (cluster labels)
    labels = db.labels_

    # Filtrar los puntos que no son outliers (label != -1)
    no_outliers_mask = labels != -1
    df_filtered = df[no_outliers_mask].copy()

    # Calcular Background_df como las diferencias entre el DataFrame original y el filtrado
    Background_df = pd.concat([df, df_filtered], sort=False).drop_duplicates(keep=False)

    return df_filtered, Background_df, eps, min_samples, labels

# Ejemplo de uso:
# df = pd.read_csv('path_to_data.csv')  # Cargar tus datos originales en un DataFrame
# df_filtered, background = apply_dbscan_filter(df)


# Función adicional para obtener clusters sin outliers
def Points_no_outliers(model, data):
    """
    Obtiene los puntos que no son outliers según el modelo de DBSCAN.

    :param model: El modelo entrenado de DBSCAN.
    :param data: Coordenadas originales para identificar puntos no considerados outliers.
    :return: Diccionario con las coordenadas x, y y los colores de los puntos no outliers.
    """
    x_out = []
    y_out = []
    colors = []

    xs = data[:, 0]
    ys = data[:, 1]

    for i in range(len(model.labels_)):
        if model.labels_[i] != -1:
            x_out.append(xs[i])
            y_out.append(ys[i])
            colors.append(model.labels_[i])

    return {'x_out': x_out, 'y_out': y_out, 'colors': colors}


# In[ ]:




