import os
import numpy as np
import h5py

deep_win = 1024

""" CARGAR DATA PARA PRUEBAS """

""" Load DAS data """
with h5py.File('data/DAS_data.h5', "r") as f:
    Nch, Nt = f["strainrate"].shape
    split = int(0.45 * Nt) #incluye todo menos train data
    data = f["strainrate"][:, split:].astype(np.float32)

# se normaliza cada trace respecto a su desviación estandar
data /= data.std()

Nwin = data.shape[1] // deep_win
# Total number of time samples to be processed
Nt_deep = Nwin * deep_win #


data_split = np.stack(np.split(data[:, :Nt_deep], Nwin, axis=-1), axis=0)
data_split = np.stack(data_split, axis=0)
data_split = np.expand_dims(data_split, axis=-1)
# Buffer for impulses
batch_size = 1 # PARA TENER SOLO UN DATO EN 1 BATCH

x = np.zeros_like(data_split)
N = data_split.shape[0] // batch_size
r = data_split.shape[0] % batch_size
for i in range(N):
    n_slice = slice(i * batch_size, (i + 1) * batch_size)
    x_i = data_split[n_slice]
    x[n_slice] = x_i
# If there is some residual chunk: process that too
if r > 0:
    n_slice = slice((N-1 + 1) * batch_size, None)
    x_i = data_split[n_slice]
    x[n_slice] = x_i

indices = []
with open('utils/datos_buenos.txt', 'r') as archivo:
    # Itera sobre cada línea en el archivo
    for linea in archivo:
        elementos = linea.split(',')
        i = elementos[0]
        image_index = int(i)
        indices.append(image_index)

datos_a_guardar = x[indices]
np.save('data/datos_seleccionados.npy', datos_a_guardar)