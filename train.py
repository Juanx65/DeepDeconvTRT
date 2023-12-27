import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
from utils.models import UNet
from torch.utils.data import TensorDataset, DataLoader
import argparse
import matplotlib.pyplot as plt

import h5py
from utils.datagenerator import DataGenerator
import os

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

def train(opt):
    """ Variables necesarias """
    cwd = os.getcwd()
    datadir = os.path.join(cwd, opt.data_dir)
    data_file = os.path.join(datadir, "DAS_data.h5")

    """ Load DAS data """
    # DAS_data.h5 -> datos para leer (1.5GB) strain rate 
    with h5py.File(data_file, "r") as f:
         # Nch : numero de canales, Nt = largo de muestras 8.626.100
        Nch, Nt = f["strainrate"].shape
        split = int(0.9 * Nt) #90% datos para entrenamiento y validación
        data = f["strainrate"][:, 0:split].astype(np.float32)
    # se normaliza cada trace respecto a su desviación estandar
    data /= data.std()
    Nch, Nt = data.shape

    window = opt.deep_win
    samples_per_epoch = 1000 # data que se espera por epoca al entrenar
    batches = opt.batch_size
    _, Nt_int = data.shape
    split = int(0.5 * Nt_int)

    train_raw_data = data[:,0:split]
    val_raw_data = data[:,split:]

    train_data = DataGenerator(train_raw_data, window, samples_per_epoch)
    val_data = DataGenerator(val_raw_data, window, samples_per_epoch)

    print("train data: ", train_data[0].size())

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False)

    """ Load impulse response """
    kernel = np.load(os.path.join(datadir,"kernel.npy"))
    kernel = kernel[..., 300:551]
    kernel = kernel[::-1].copy() # invertido temporalmente
    kernel = kernel / kernel.max()
    kernel = torch.from_numpy(kernel).float()
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Añade las dimensiones de batch y canal
    kernel = kernel.to(device)
    kernel = kernel.expand(opt.batch_size, -1, -1, -1)
    kernel = kernel.view(opt.batch_size, 1, 1, -1)

    # Cargar modelo
    model = UNet()
    model.to(device)
    summary(model)
    loss_function = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate,weight_decay=opt.l1_lambda)

    # Entrenamiento
    valid_loss_min = np.Inf
    n_epoch = opt.epochs
    for epoch in range(n_epoch):
        train_loss = 0.0
        valid_loss = 0.0
        last_batch_index_train = 1
        for batch_idx, x_batch in enumerate(train_loader):
        
            if x_batch.size(0) != opt.batch_size:
                break
    
            x_batch = x_batch.to(device)
            x_batch = x_batch.unsqueeze(1)

            y_pred = model(x_batch) # Forward

            ##########################################################################################

            x_hat = F.conv2d(y_pred, kernel, padding='same', stride=1)
            x_pred = x_hat[:, :, :opt.deep_win, :]

            ###########################################################################################

            loss = loss_function(x_pred, x_batch)
            l1_loss = l1_penalty(model.parameters())
            loss = loss + opt.l1_lambda * l1_loss

            # Backward y optimización
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()

            train_loss += loss.item()*x_batch.size(0)
            last_batch_index_train = batch_idx
        last_batch_index_val = 1
        for batch_idx, x_batch in enumerate(valid_loader):
            if x_batch.size(0) != opt.batch_size:
                break

            x_batch = x_batch.to(device)
            x_batch = x_batch.unsqueeze(1)
            
            y_pred = model(x_batch)

            ##########################################################################################
            
            x_hat = F.conv2d(y_pred, kernel, padding='same', stride=1)
            x_pred = x_hat[:, :, :opt.deep_win, :]
            
            ###########################################################################################
            
            loss = loss_function(x_pred, x_batch)
            #l1_loss = l1_penalty(model.parameters())
            #loss = loss + opt.l1_lambda * l1_loss

            valid_loss +=  loss.item()*x_batch.size(0)
            last_batch_index_val = batch_idx
        
        train_loss = train_loss / (last_batch_index_train*x_batch.size(0))
        valid_loss = valid_loss / (last_batch_index_val*x_batch.size(0))
        if( valid_loss < valid_loss_min):
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
            torch.save(model, opt.weights)
            valid_loss_min = valid_loss
        if (epoch+1) % 1 == 0:
            print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch + 1, n_epoch, train_loss,valid_loss))

def l1_penalty(parameters):
    return sum(p.abs().sum() for p in parameters)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 128, type=int,help='batch size to train')
    parser.add_argument('--epochs', default = 200, type=int,help='epoch to train')
    parser.add_argument('--learning_rate', default = 0.001, type=float, help='learning rate')
    parser.add_argument('--data_dir', default = "data",type=str,help='dir to the dataset')
    parser.add_argument('--weights', default = 'weights/best.pth', type=str, help='directorio y nombre de archivo de donse se guardara el mejor peso entrenado')
    parser.add_argument('--deep_win', default = 1024,type=int,help='Number of samples per chunk')
    parser.add_argument('--l1_lambda', default = 1e-3, type=float, help='Weight decay (default: 1e-4)')
    
    opt = parser.parse_args()
    return opt

def main(opt):
    train(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)