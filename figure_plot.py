import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scipy.spatial import distance

from models_epi import Model
from utils_epi import loss_function, load_mat, show_image, CombinedDataset, show_image_group, plot_ternary
from hom_FE import StructuralFE

### set plot text font
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

if __name__ == '__main__':
    torch.manual_seed(42)
    cuda = False
    device = torch.device("cuda" if cuda else "cpu")
    im_x = 50
    im_y = 50 
    num_properties = 5  # Number of properties in the dataset
    modes1 = 10
    modes2 = 6
    model_type = 'Freq_FNO'

    plot_data_dir = './plot_data'
    figure_dir = './figures'

    hidden_dim = 64
    latent_dim = 24

    dataset_name = '_Chen_data_9_1_'
    experiment_name = model_type+dataset_name+ str(latent_dim)+"_"+str(hidden_dim)

    ### load confidence_vs_designed_error data
    confidence_vs_designed_error = np.loadtxt(osp.join(plot_data_dir, experiment_name+'_confidence_vs_designed_error.txt'),delimiter=',').astype(np.float32)
    confidence_vs_min_dist = np.loadtxt(osp.join(plot_data_dir, experiment_name+'_confidence_vs_min_dist.txt'),delimiter=',').astype(np.float32)

    ### plot mini_dist vs designed error
    plt.figure(figsize=(6, 4))
    plt.scatter(confidence_vs_min_dist[:,0], confidence_vs_designed_error[:,0], s=1)
    plt.xlabel('Minimum distance to training data')
    plt.ylabel('Designed error')
    plt.title('Minimum distance to training data vs Designed error')
    plt.grid(True, which="both", ls="--")
    plt.savefig(osp.join(figure_dir, experiment_name+'_min_dist_vs_designed_error.jpg'), dpi=500)
    plt.close()

    ### plot confidencs vs designed error
    plt.figure(figsize=(6, 4))
    plt.scatter(confidence_vs_designed_error[:,0], confidence_vs_designed_error[:,1], s=1)
    plt.xlabel('Designed error')
    plt.ylabel('Data familiarity')
    plt.xlim(0,0.3)
    plt.grid(True, which="both", ls="--")
    plt.savefig(osp.join(figure_dir, experiment_name+'_confidence_vs_designed_error.jpg'), dpi=500)
    plt.close()

    ### plot confidence vs minimum distance
    plt.figure(figsize=(6, 4))
    plt.scatter(confidence_vs_min_dist[:,0], confidence_vs_min_dist[:,1], s=1)
    plt.xlabel('Minimum distance to training data')
    plt.ylabel('Data familiarity')
    plt.xlim(0,2)
    plt.ylim(0,1)
    plt.grid(True, which="both", ls="--")
    plt.savefig(osp.join(figure_dir, experiment_name+'_confidence_vs_min_dist.jpg'), dpi=500)
    plt.close()

    ### plot confidence vs designed error using line approaximation with standard deviation
    sorted_indices = np.argsort(confidence_vs_designed_error[:,1])
    sorted_confidence = confidence_vs_designed_error[sorted_indices, 1]
    sorted_designed_error = confidence_vs_designed_error[sorted_indices, 0]
    window_size = 300
    averaged_confidence = np.convolve(sorted_confidence, np.ones(window_size)/window_size, mode='valid')
    averaged_designed_error = np.convolve(sorted_designed_error, np.ones(window_size)/window_size, mode='valid')
    std_designed_error = np.array([np.std(sorted_designed_error[max(0, i - window_size // 2):min(len(sorted_designed_error), i + window_size // 2)]) for i in range(len(averaged_designed_error))])
    plt.figure(figsize=(6, 4))
    plt.plot(averaged_confidence, averaged_designed_error, color='blue', label='Averaged Designed Error')
    plt.fill_between(averaged_confidence, averaged_designed_error - std_designed_error, averaged_designed_error + std_designed_error, color='blue', alpha=0.2, label='Standard Deviation')
    plt.xlabel('Data familiarity')
    plt.ylabel('Designed Property Error')
    plt.xlim(0,1)
    plt.ylim(0,0.15)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(osp.join(figure_dir, experiment_name+'_confidence_vs_designed_error_line.jpg'), dpi=500)
    plt.close() 

    ### plot confidence vs minimum distance using line approximation with standard deviation
    sorted_indices = np.argsort(confidence_vs_min_dist[:,1])
    sorted_confidence = confidence_vs_min_dist[sorted_indices, 1]
    sorted_min_dist = confidence_vs_min_dist[sorted_indices, 0]
    window_size = 300
    averaged_confidence = np.convolve(sorted_confidence, np.ones(window_size)/window_size, mode='valid')
    averaged_min_dist = np.convolve(sorted_min_dist, np.ones(window_size)/window_size, mode='valid')
    std_min_dist = np.array([np.std(sorted_min_dist[max(0, i - window_size // 2):min(len(sorted_min_dist), i + window_size // 2)]) for i in range(len(averaged_min_dist))])
    plt.figure(figsize=(6, 4))
    plt.plot(averaged_confidence, averaged_min_dist, color='blue', label='Averaged Minimum Distance')
    plt.fill_between(averaged_confidence, averaged_min_dist - std_min_dist, averaged_min_dist + std_min_dist, color='blue', alpha=0.2, label='Standard Deviation')
    plt.xlabel('Data familiarity')
    plt.ylabel('Minimum Distance to Training Data')
    plt.xlim(0,1)
    plt.ylim(0,2)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(osp.join(figure_dir, experiment_name+'_confidence_vs_min_dist_line.jpg'), dpi=500)
    plt.close()

    ### plot minimum distance vs designed error using line approximation with standard deviation
    sorted_indices = np.argsort(confidence_vs_min_dist[:,0])
    sorted_min_dist = confidence_vs_min_dist[sorted_indices, 0]
    sorted_designed_error = confidence_vs_designed_error[sorted_indices, 0]
    window_size = 200
    averaged_min_dist = np.convolve(sorted_min_dist, np.ones(window_size)/window_size, mode='valid')
    averaged_designed_error = np.convolve(sorted_designed_error, np.ones(window_size)/window_size, mode='valid')
    std_designed_error = np.array([np.std(sorted_designed_error[max(0, i - window_size // 2):min(len(sorted_designed_error), i + window_size // 2)]) for i in range(len(averaged_designed_error))])
    plt.figure(figsize=(6, 4))
    plt.plot(averaged_min_dist, averaged_designed_error, color='blue', label='Averaged Designed Error')
    plt.fill_between(averaged_min_dist, averaged_designed_error - std_designed_error, averaged_designed_error + std_designed_error, color='blue', alpha=0.2, label='Standard Deviation')
    plt.xlabel('Minimum Distance to Training Data')
    plt.ylabel('Designed Property Error')
    plt.xlim(np.min(sorted_min_dist),np.max(sorted_min_dist))
    plt.ylim(0,0.6)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(osp.join(figure_dir, experiment_name+'_min_dist_vs_designed_error_line.jpg'), dpi=500)
    plt.close() 
