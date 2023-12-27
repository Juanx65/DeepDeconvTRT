import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)

def main(opt):
    # Nombres de las columnas basados en el formato proporcionado
    column_names = ['Date', 'Time', 'gpu', 'mclk', 'pclk', 'fb', 'bar1', 'sm', 'mem', 'enc', 'dec']

    # Cargar el archivo .txt
    df = pd.read_csv(opt.csv, delim_whitespace=True, header=None, names=column_names, comment='#')

    # Combina las columnas 'Date' y 'Time' y las convierte en datetime, maneja errores
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'], format='%Y%m%d %H:%M:%S', errors='coerce')

    # Manejar los NaT resultantes si es necesario
    # Por ejemplo, podríamos querer eliminar las filas con NaT
    df.dropna(subset=['Datetime'], inplace=True)
    
    # Configura la columna 'Datetime' como el índice del DataFrame
    df.set_index('Datetime', inplace=True)

    # Calcula la suma de 'bar1' y 'fb'
    df['bar1_fb_sum'] = df['bar1'] + df['fb']

    # Inicia el proceso de creación del gráfico
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Calcular segundos desde el inicio del registro
    start_time = df.index[0]
    df['Seconds'] = (df.index - start_time).seconds

    # Graficar 'bar1_fb_sum' y 'bar1' como áreas
    ax1.fill_between(df['Seconds'], df['bar1_fb_sum'], label='bar1+fb', color='skyblue')
    ax1.fill_between(df['Seconds'], df['bar1'], label='bar1', color='blue')

    # Establecer límites para el eje Y de memoria
    ax1.set_ylim(0, 12288)  # Asumiendo que 12288 MiB es el 100%

    # Crear un segundo eje que comparte el mismo eje X para la utilización de la GPU
    ax2 = ax1.twinx()

    # Graficar 'sm' como una línea
    ax2.plot(df['Seconds'], df['sm'], label='SM Utilization', color='green', linewidth=2)

    # Establecer límites para el eje Y de utilización de la GPU
    ax2.set_ylim(0, 101)  # Asumiendo que 100% es el máximo

    # Títulos y etiquetas
    ax1.set_title('GPU Memory Usage and SM Utilization Over Time')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Memory Usage (MB)', color='blue')
    ax2.set_ylabel('SM Utilization (%)', color='green')

    # Leyendas
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Guardar el gráfico
    if not opt.output:
        nombre_base = os.path.splitext(opt.csv)[0]
        nombre_figura = f'{nombre_base}.png'
    else:
        # Verificar si el path existe
        if not os.path.exists(opt.output):
            # Si no existe, crear los directorios necesarios
            os.makedirs(os.path.dirname(opt.output), exist_ok=True)
        nombre_figura = opt.output
    plt.savefig(nombre_figura, bbox_inches='tight')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='../outputs/gpu_usage/gpu_usage.csv', help='path to csv')
    parser.add_argument('--output', default='', help='path to output image')
 
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)