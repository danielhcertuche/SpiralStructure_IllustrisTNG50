import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial import ConvexHull, distance
from scipy.spatial.distance import cdist

def create_velocity_grid(processed_data):
    # Grid
    v_circ_min, v_circ_max = processed_data['velocity_magnitude'].min(), processed_data['velocity_magnitude'].max()
    v_esc_min, v_esc_max = processed_data['escape_velocity'].min(), processed_data['escape_velocity'].max()

    v_circ_bins = np.arange(v_circ_min, v_circ_max, 8)
    v_esc_bins = np.arange(v_esc_min, v_esc_max, 8)

    grid_data = np.zeros((len(v_circ_bins) - 1, len(v_esc_bins) - 1))
    point_density = np.zeros((len(v_circ_bins) - 1, len(v_esc_bins) - 1))

    for i, j in [(i, j) for i in range(len(v_circ_bins) - 1) for j in range(len(v_esc_bins) - 1)]:
        mask = ((processed_data['velocity_magnitude'] >= v_circ_bins[i]) &
                (processed_data['velocity_magnitude'] < v_circ_bins[i + 1]) &
                (processed_data['escape_velocity'] >= v_esc_bins[j]) &
                (processed_data['escape_velocity'] < v_esc_bins[j + 1]))

        points_in_cell = processed_data[mask]['velocity_magnitude']
        point_density[i, j] = len(points_in_cell)
        if len(points_in_cell) > 0:
            grid_data[i, j] = np.mean(points_in_cell)

    weighted_avg_velocity_per_column = [
        np.average(grid_data[:, j], weights=point_density[:, j]) if np.sum(point_density[:, j]) > 0 else np.nan
        for j in range(grid_data.shape[1])
    ]

    weighted_avg_velocity_per_column = np.array(weighted_avg_velocity_per_column)
    v_esc_midpoints = (v_esc_bins[:-1] + v_esc_bins[1:]) / 2
    max_avg_velocity = np.nanmax(weighted_avg_velocity_per_column)

    return grid_data, v_esc_midpoints, weighted_avg_velocity_per_column, max_avg_velocity, v_circ_min, v_circ_max, v_esc_min, v_esc_max

def filter_data_by_velocity(processed_data, max_avg_velocity, v_esc_midpoints, weighted_avg_velocity_per_column, percentile_range=(10, 99)):
    # Filtrar las partículas que cumplen con el criterio del 70% de la velocidad máxima promedio
    threshold = 0.7 * max_avg_velocity
    filtered_data = processed_data[processed_data['velocity_magnitude'] > threshold]

    # Filtrar según percentiles alrededor de max_avg_velocity en el eje x (escape_velocity)
    percentile_20 = np.percentile(filtered_data['escape_velocity'], percentile_range[0])
    percentile_80 = np.percentile(filtered_data['escape_velocity'], percentile_range[1])
    filtered_data = filtered_data[(filtered_data['escape_velocity'] >= percentile_20) &
                                  (filtered_data['escape_velocity'] <= percentile_80)]

    return filtered_data


def apply_convex_hull_filter(rotated_vectors_position, filtered_data, rotated_data, idx):
    # Aplicar Convex Hull a los datos filtrados
    rotated_vectors_filtered = rotated_vectors_position[filtered_data.index]
    x = rotated_vectors_filtered[:, 0]
    y = rotated_vectors_filtered[:, 1]
    z = rotated_vectors_filtered[:, 2]
    points_3d = np.vstack((x, y, z)).T

    # Elegir un número pequeño de puntos aleatorios por coordenada
    n_points_per_coord = min(10000, len(points_3d))
    sample_indices = np.random.choice(points_3d.shape[0], n_points_per_coord, replace=True)
    sample_points = points_3d[sample_indices]

    # Calcular la envolvente convexa
    hull_3d = ConvexHull(sample_points)

    # Verificar si los puntos están dentro del Convex Hull
    def puntos_prueba_dentro(hull, puntos):
        puntos_transformados = hull.equations[:, :-1].dot(puntos.T) + hull.equations[:, -1][:, np.newaxis]
        esta_dentro = np.all(puntos_transformados <= 0, axis=0)
        return esta_dentro

    # Definir los puntos en 3D
    x_total = rotated_vectors_position[:, 0]
    y_total = rotated_vectors_position[:, 1]
    z_total = rotated_vectors_position[:, 2]
    points_3d_datos = np.vstack((x_total, y_total, z_total)).T

    # Verificar qué puntos están dentro del Convex Hull del TOTAL de datos
    esta_dentro_hull = puntos_prueba_dentro(hull_3d, points_3d_datos)

    # Calcular las distancias desde los puntos de la muestra a los vértices de la envolvente convexa
    distances = cdist(sample_points, sample_points[hull_3d.vertices])
    max_distance = np.max(distances)
    
    all_data_points = rotated_data
    # Extraer las propiedades de los puntos dentro del Convex Hull
    datos_dentro_hull = all_data_points[esta_dentro_hull]
    column_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'lxvel', 'lyvel', 'lzvel', 'Potential', 'U', 'rho']
    df_hull_data = pd.DataFrame(datos_dentro_hull, columns=column_names)

    # Guardar como CSV para preservar la estructura y facilitar la lectura posterior
    df_hull_data.to_csv(f'DATA/processed/halo_{idx}_datos_dentro_hull.csv', index=False)

    return hull_3d, points_3d_datos, esta_dentro_hull, max_distance, sample_points
