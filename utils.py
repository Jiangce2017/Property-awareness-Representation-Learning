import torch
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

def loss_function_vae(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD