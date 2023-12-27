import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DataGenerator(Dataset):
    def __init__(self, data, window_size, num_samples):
        """
        Args:
            data (numpy.ndarray): Datos en bruto para el entrenamiento o la validación.
            window_size (int): Tamaño de la ventana de muestras a generar.
            num_samples (int): Número de muestras por época.
        """
        self.data = data
        self.window_size = window_size
        self.num_samples = num_samples

    def __len__(self):
        """
        Denota el número total de muestras.
        """
        return self.num_samples

    def __getitem__(self, index):
        """
        Genera una muestra de los datos.
        """
        # Selección aleatoria de un punto de inicio para la ventana de datos
        start_idx = np.random.randint(0, self.data.shape[1] - self.window_size)
        # Extracción de la ventana de datos
        sample = self.data[:, start_idx:start_idx + self.window_size]
        return torch.from_numpy(sample)

# Uso de la clase DataGenerator
# train_data = DataGenerator(train_raw_data, window, samples_per_epoch)
# val_data = DataGenerator(val_raw_data, window, samples_per_epoch)
