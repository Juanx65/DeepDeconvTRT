################################################################################
""" IMPORTS SECTION """
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils.models import UNet
import argparse
from utils.utils import *
import time
from statistics import mean 
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)


################################################################################
# Global consts
default_kernel = 'data/kernel.npy' 
default_data = 'data/DAS_data.h5'

def main(opt):

    """ LOAD KERNEL """
    # Verify if file exists
    if not(os.path.exists(opt.kernel)):
        print("The kernel file <{}> does not exists!".format(opt.kernel))
        exit(1)
    kernel = np.load(opt.kernel)
    kernel = kernel[..., 300:551]
    kernel = kernel[::-1].copy() # invertido temporalmente
    kernel = kernel / kernel.max() # Kernel normalization
    kernel = torch.from_numpy(kernel).float()
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Añade las dimensiones de batch y canal
    kernel = kernel.to(device)
    kernel = kernel.expand(1, -1, -1, -1)
    kernel = kernel.view(1, 1, 1, -1)
    
    """ Init Deep Learning model """
    if opt.compare or not opt.trt:
        model = UNet()
        model = torch.load(opt.weights)
    if opt.trt:
        from utils.engine import TRTModule #if not done here, unable to train
        current_directory = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(current_directory,opt.engine)
        Engine = TRTModule(engine_path,device)
        Engine.set_desired(['outputs'])
        if opt.compare:
            engine_path = os.path.join(current_directory,opt.engine2)
            Engine2 = TRTModule(engine_path,device)
            Engine2.set_desired(['outputs'])
            Engine.to(device).eval()
            Engine2.to(device).eval()
        else:
            model = Engine
    model.to(device).eval()

    if opt.compare:
        compare(opt,kernel, model, Engine, Engine2, 1, opt.r_tol )
    else:
        eval(opt, kernel, model)


def eval(opt, kernel, model):

    deep_win = opt.deep_win

    """ CARGAR DATA PARA PRUEBAS """

    """ Load DAS data """

    if opt.numpy_data:
        data = np.load('data/datos_seleccionados.npy')
    else: 
        if not(os.path.exists(opt.data)):
            print("Data file {} does not exists!".format(opt.data))
            exit(1)
        with h5py.File(opt.data, "r") as f:
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
        data = x
    
    # Itera sobre cada línea en el archivo
    total_time = 0
    max_time = 0
    for index_x, x in enumerate(data):
  
        input_data = x[None,:,:]
        input_data = torch.from_numpy(input_data)

        start_time = time.time()

        input_data = input_data.to(device)
        input_data = input_data.unsqueeze(0)
        input_data = input_data.squeeze(-1)
    
        y_hat = model(input_data)

        end_time = time.time()
        iteration_time = end_time - start_time
        total_time += iteration_time
        if iteration_time > max_time:
            max_time = iteration_time

        ##########################################################################################
        
        x_hat = F.conv2d(y_hat, kernel, padding='same', stride=1)
        x_hat = x_hat[:, :,:opt.deep_win,:]

        ###########################################################################################
        
        x_hat = x_hat.view(24,1024)
        y_hat = y_hat.view(24,1024)

        x_hat = x_hat.cpu().detach().numpy()
        y_hat = y_hat.cpu().detach().numpy()

        
        # GRAFICAR LOS RESULTADOS # se toman muchisimo tiempo
        if opt.plot:
            samp = 80.
            t = np.arange(x_hat.shape[1]) / samp

            f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
            ax1.set_title('S')
            ax2.set_title('E_hat')
            ax3.set_title('S_hat')

            f.suptitle('DATA'+ str(index_x), fontsize=16)
            #subplot1: origina
            for i, wv in enumerate(x):
                ax1.plot( t, wv - 8 * i, "tab:orange",linewidth=2.5)
            plt.tight_layout()
            plt.grid()

            #subplot2: x_hat-> estimación de la entrada (conv kernel con la salida)
            for i, wv in enumerate(y_hat):
                ax2.plot(t,(10*wv - 8 * i), "tab:red", linewidth=2.5)
            plt.tight_layout()
            plt.grid()

            #subplot3: y_hat->
            for i, wv in enumerate(x_hat):
                ax3.plot(t,wv - 8 * i, c="k",linewidth=2.5)
            plt.tight_layout()
            plt.grid()

            #plt.savefig("figures/multi_cars_impulse.pdf")
            plt.grid()
            #plt.show()
            nombre_archivo = f'outputs/img_results/{index_x}_{opt.network}.png'
            plt.savefig(nombre_archivo)
            plt.close() 
       
    total_iterations = len(data)
    average_time = total_time / total_iterations

    print(f"Tiempo promedio {opt.network}: {average_time*1000} ms")
    print(f"Tiempo máximo {opt.network}: {max_time*1000} ms")
    print("datos procesados: ", len(data))


def compare(opt, kernel, model, Engine_fp32,Engine_fp16, batch_size,rtol):
    model.eval()
    Engine_fp32.eval()
    Engine_fp16.eval()

    top_n = 1
    disagreements_fp32 = np.zeros(top_n,dtype=np.int32)  # Track the number of disagreements for top1 to top5
    disagreements_fp16 = np.zeros(top_n,dtype=np.int32) 

    mae_errors_fp32 = np.zeros(top_n,dtype=np.float64)
    mae_errors_fp16 = np.zeros(top_n,dtype=np.float64)

    num_batches = 5000

    closeness_count_fp32 = 0
    closeness_count_fp16 = 0

    MAES_fp32 = []
    MSES_fp32 = []
    MAES_fp16 = []
    MSES_fp16 = []
    for i in range(num_batches):
        torch.manual_seed(i)
        input = torch.rand(batch_size, 1, 24, 1024) # generamos un input random [0,1)

        input = input.to(device)
        
        if input.size(0) != batch_size:
            print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
            break

        with torch.no_grad():
            # compute output
            output_vanilla = model(input)
            output_fp32 = Engine_fp32(input)
            output_fp16 = Engine_fp16(input)

        close_values_fp32 = torch.isclose(output_vanilla, output_fp32, rtol=rtol).sum().item()
        close_values_fp16 = torch.isclose(output_vanilla, output_fp16, rtol=rtol).sum().item()

        closeness_count_fp32 += close_values_fp32
        closeness_count_fp16 += close_values_fp16

        MAES_fp32.append(F.l1_loss(output_vanilla, output_fp32).item())
        MSES_fp32.append(F.mse_loss(output_vanilla, output_fp32).item())
        MAES_fp16.append(F.l1_loss(output_vanilla, output_fp16).item())
        MSES_fp16.append(F.mse_loss(output_vanilla, output_fp16).item())

    total_values = output_vanilla.size(0)* output_vanilla.size(1)  * output_vanilla.size(2) * output_vanilla.size(3) * num_batches
    closeness_percentage_fp32 = (closeness_count_fp32 / total_values) * 100
    closeness_percentage_fp16 = (closeness_count_fp16 / total_values) * 100
    print(" ")
    # Print header
    print("| Vanilla VS | equality (%) |")
    print("|------------|--------------|")
    print("| TRT fp32 | {:.2f} |".format(closeness_percentage_fp32))
    print("| TRT fp16 | {:.2f} |".format(closeness_percentage_fp16))
    
    print(" ")

    print("| MAE fp32 | MAE fp16 | MSE fp32 | MSE fp16 |")
    print("|----------|----------|----------|----------|")

    print("| {:<8} | {:<8} | {:<8} | {:<8} |".format(
        "{:.8f}".format(mean(MAES_fp32)),
        "{:.8f}".format(mean(MAES_fp16)),
        "{:.8f}".format(mean(MSES_fp32)),
        "{:.8f}".format(mean(MSES_fp16))
    ))
    print(" ")
   
    return


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=default_data,type=str,help='Dataset to load.')
    parser.add_argument('--weights',default = 'weights/best.pth', type=str,help='Load weights path.')
    parser.add_argument('--engine',default = 'weights/best.engine', type=str,help='Load weights path.')
    parser.add_argument('--engine2',default = 'weights/best_fp16.engine', type=str,help='Load weights path.')
    parser.add_argument('--deep_win', default = 1024,type=int,help='Number of samples per chunk.')
    parser.add_argument('--r_tol', default = 1e-2,type=float,help='relative tolerance to compare with isclose.')
    parser.add_argument('--kernel', default = default_kernel, help='Indicates which kernel to use. Receives a <npy> file.')
    parser.add_argument('--network', default = 'vanilla', help='Indicates the name of the network to save de image result.')
    parser.add_argument('-trt','--trt', action='store_true',help='evaluate model on validation set al optimizar con tensorrt')
    parser.add_argument('-numpy_data','--numpy_data', action='store_true',help='add if using the numpy data in data/datos_seleccionados.npy')
    parser.add_argument('-plot','--plot', action='store_true',help='to save the result plots in folder output/img_results')
    parser.add_argument('-compare','--compare', action='store_true',help='compare the vanilla model with the optimized engine based on the isclose value and other metrics.')
    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
