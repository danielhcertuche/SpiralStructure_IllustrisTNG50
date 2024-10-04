
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def visualize_initial_data(ax, grid_data, v_esc_min, v_esc_max, v_circ_min, v_circ_max, processed_data, v_esc_midpoints, weighted_avg_velocity_per_column, max_avg_velocity, filtered_data, file_name):
    """
    Visualiza los datos iniciales de velocidades y puntos filtrados en un gráfico.

    Parameters:
    - ax: El eje sobre el que se realizará la gráfica.
    - grid_data: Datos de la grilla de velocidad.
    - v_esc_min, v_esc_max: Mínimo y máximo de velocidad de escape.
    - v_circ_min, v_circ_max: Mínimo y máximo de velocidad circular.
    - processed_data: DataFrame con los datos procesados.
    - v_esc_midpoints: Puntos medios de la velocidad de escape.
    - weighted_avg_velocity_per_column: Promedio ponderado de la velocidad circular.
    - max_avg_velocity: Máxima velocidad circular promedio.
    - filtered_data: DataFrame con los datos filtrados.
    - file_name: Nombre del archivo, usado para el título del gráfico.
    """

    ax.imshow(grid_data, origin='lower', aspect='auto', extent=[v_esc_min, v_esc_max, v_circ_min, v_circ_max], cmap='viridis')
    ax.scatter(processed_data['escape_velocity'], processed_data['velocity_magnitude'], color='red', s=1, alpha=0.45, label='Real Data Points')
    ax.axhline(y=max_avg_velocity, color='white', linestyle='--', label=f'Max Avg Velocity = {max_avg_velocity:.2f}')
    ax.scatter(v_esc_midpoints, weighted_avg_velocity_per_column, color='black', s=40, label='Average Velocity Per Column')
    
    # Agregar datos filtrados
    ax.scatter(filtered_data['escape_velocity'], filtered_data['velocity_magnitude'], color='blue', s=1, alpha=0.45, label='Filtered Data')
    ax.scatter(v_esc_midpoints, weighted_avg_velocity_per_column, color='black', s=40, label='Average Velocity Per Column')
    
    ax.set_title(f'{file_name}')
    ax.set_xlabel(r'$V_{esc}$')
    ax.set_ylabel(r'$V_{circ}$')
    ax.legend(loc=2)
    ax.grid(True)


def visualize_convex_hull(hull_3d, sample_points, points_3d_datos, esta_dentro_hull, file_path):
    # Visualizar el Convex Hull y los puntos de prueba
    fig = plt.figure(figsize=(13, 10))

    # Subplot 1: Vista desde el frente
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.add_collection(Poly3DCollection(sample_points[hull_3d.simplices], alpha=0.1, linewidths=0.5, edgecolors='k', linestyle='--'))
    ax1.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], color='blue', s=0.03)
    ax1.set_title(f'Convex Hull 3D - {file_path}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # Subplot 2: Vista desde arriba
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.add_collection(Poly3DCollection(sample_points[hull_3d.simplices], alpha=0.1, linewidths=0.5, edgecolors='k', linestyle='--'))
    ax2.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], color='blue', s=0.03)
    ax2.scatter(points_3d_datos[esta_dentro_hull][:, 0], points_3d_datos[esta_dentro_hull][:, 1], points_3d_datos[esta_dentro_hull][:, 2], color='green', s=0.10)
    ax2.scatter(points_3d_datos[~esta_dentro_hull][:, 0], points_3d_datos[~esta_dentro_hull][:, 1], points_3d_datos[~esta_dentro_hull][:, 2], color='red', s=0.50)
    ax2.set_title('Puntos de Prueba')
    ax2.view_init(220, 360)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    # Subplot 3: Vista desde el lado
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.add_collection(Poly3DCollection(sample_points[hull_3d.simplices], alpha=0.1, linewidths=0.5, edgecolors='k', linestyle='--'))
    ax3.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], color='blue', s=0.03)
    ax3.scatter(points_3d_datos[esta_dentro_hull][:, 0], points_3d_datos[esta_dentro_hull][:, 1], points_3d_datos[esta_dentro_hull][:, 2], color='green', s=0.10)
    ax3.scatter(points_3d_datos[~esta_dentro_hull][:, 0], points_3d_datos[~esta_dentro_hull][:, 1], points_3d_datos[~esta_dentro_hull][:, 2], color='red', s=0.50)

