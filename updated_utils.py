# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import h5py

class CombinedDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
    
    def __len__(self):
        return min(len(self.input_data), len(self.output_data))
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

def load_mat(filename):
    with h5py.File(filename, 'r') as f:
        data = {}
        for k, v in f.items():
            data[k] = v[:]  # Load data into memory
    return data
    
def show_image(x):
        fig = plt.figure()
        cmap = 'Greens'
        plt.imshow(x.cpu().numpy(),cmap=cmap)
        plt.show()

def loss_function(y_pred, y_true):
    loss = torch.sum((y_pred-y_true)**2)
    return loss


def loss_function_vae(x, x_hat, material_pred, y_true, log_var, kl_weight=1.0):
    x = torch.clamp(x, 1e-7, 1 - 1e-7)
    x_hat = torch.clamp(x_hat, 1e-7, 1 - 1e-7)

    recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')
    kl = -0.5 * torch.sum(1 + log_var - log_var.exp()) / x.size(0)
    mse = nn.functional.mse_loss(material_pred, y_true, reduction='mean')

    print("recon_loss: {}, kl: {}, mse: {}".format(recon_loss, kl,mse ))

    return recon_loss + kl_weight * kl + mse
