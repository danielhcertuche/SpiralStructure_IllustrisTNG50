import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from ROUTINES.data_preprocessing.rotation_handler import RotationHandler

class TNG50DataProcessor:
    def __init__(self, file_path=None, data=None):
        """
        Inicializa la clase con un archivo o directamente con los datos.
        :param file_path: Ruta del archivo de datos.
        :param data: DataFrame con los datos ya cargados (opcional).
        """
        if file_path:
            self.file_path = file_path
            self.chunk_size = 102609 // 9
            self.load_data()
        elif data is not None:
            # Si se proporciona un DataFrame, se utiliza directamente
            self.accumulated_data = data.values
        else:
            raise ValueError("Debe proporcionar 'file_path' o 'data'.")
    

    def load_data(self):
        chunks = pd.read_csv(self.file_path, sep=' ', names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'lxvel', 'lyvel', 'lzvel', 'Potential', 'U', 'rho'], header=None, chunksize=self.chunk_size)
        self.accumulated_data = np.concatenate([chunk.values for chunk in chunks])

    def calculate_center_of_mass_new(self, value_energy=2.63):
        Log_energy = np.log10(self.accumulated_data[:, 10])
        avg_energy = np.mean(Log_energy)
        if value_energy == 1:
            value_energy = avg_energy
        energy_filter = Log_energy < value_energy
        self.filtered_data = self.accumulated_data[energy_filter]
        return np.sum(self.filtered_data[:, :3], axis=0) / len(self.accumulated_data)

    def calculate_total_angular_momentum(self):
        center_of_mass = self.calculate_center_of_mass_new()
        r_i = self.filtered_data[:, :3] - center_of_mass
        v_i = self.filtered_data[:, 3:6] - self.calculate_velocity_of_mass()
        return np.sum(np.cross(r_i, v_i), axis=0)

    def calculate_velocity_of_mass(self):
        return np.mean(self.filtered_data[:, 3:6], axis=0)
    
    def apply_rotation_auxiliar(self, matrix, vectors, block_size=5000):
        """
        Aplica la rotación por bloques a los datos, para evitar problemas de memoria.
        :param matrix: Matriz de rotación.
        :param vectors: Vectores de posiciones o velocidades a rotar.
        :param block_size: Tamaño del bloque de datos que se procesará a la vez.
        :return: Vectores rotados.
        """
        rotated_data = np.zeros_like(vectors)
        num_vectors = vectors.shape[0]

        # Procesar los datos en bloques
        for i in range(0, num_vectors, block_size):
            end = min(i + block_size, num_vectors)
            rotated_data[i:end] = np.dot(vectors[i:end], matrix.T)

        return rotated_data
    
    def apply_rotation(self, matrix, vectors):
        # Aplicar la matriz de rotación a todas las posiciones
        return np.dot(vectors, matrix.T)
    
    
    def align_with_z_axis(self, total_angular_momentum):
        
        # Normalizar el vector de momento angular
        norm_angular_momentum = total_angular_momentum / np.linalg.norm(total_angular_momentum)
        
        # Vector Z unitario
        z_axis = np.array([0, 0, 1])
        
        # Calcular la rotación necesaria para alinear el momento angular con el eje Z
        rotation_vector = np.cross(norm_angular_momentum, z_axis)  # Producto cruzado entre el momento angular y el eje Z
        sin_angle = np.linalg.norm(rotation_vector)
        cos_angle = np.dot(norm_angular_momentum, z_axis)
        rotation_angle = np.arctan2(sin_angle, cos_angle)
        
        # Si el ángulo es cero, no se necesita rotación
        if sin_angle != 0:
            rotation_axis = rotation_vector / sin_angle
            rotation_matrix = Rotation.from_rotvec(rotation_angle * rotation_axis).as_matrix()
        else:
            rotation_matrix = np.identity(3)
        
        return rotation_matrix
    
    def visualize_rotated_vector(self):
        # Calcular el momento angular total
        total_angular_momentum = self.calculate_total_angular_momentum()
        rotation_matrix = self.align_with_z_axis(total_angular_momentum)
        rotated_data_positions = self.apply_rotation(rotation_matrix, self.filtered_data[:, :3])
        rotated_data_velocities = self.apply_rotation(rotation_matrix, self.filtered_data[:, 3:6])
        rotated_data_total = np.concatenate([rotated_data_positions, rotated_data_velocities, self.filtered_data[:, 6:]], axis=1)
        return rotated_data_positions,rotated_data_total

    
    
    
    
