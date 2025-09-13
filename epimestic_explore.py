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

def compute_error(index, x_hat_list, target_prop_list, hom_FE_solver):
    x_hat_Q = hom_FE_solver.solve(x_hat_list[index].permute(1,0).reshape(im_x*im_y))
    x_hat_Q = x_hat_Q.cpu().detach().numpy()
    x_hat_density = np.mean(x_hat_list[index].cpu().detach().numpy())
    designed_prop = np.array([x_hat_Q[0,0], x_hat_Q[0,1], x_hat_Q[1,1], x_hat_Q[2,2], x_hat_density])
    target_prop = target_prop_list[index,:num_properties]
    design_err = np.mean((designed_prop - target_prop)**2)
    return design_err

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

    lattice_data_dir = '/scratch/jc14407/datasets/Chen/trainning_test_9_1'
    # dataset_path = osp.join(lattice_data_dir, 'rho_data.csv')
    # property_path = osp.join(lattice_data_dir, 'property_data.csv')
    results_dir = './results'
    plot_data_dir = './plot_data'
    figure_dir = './figures'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(plot_data_dir):
        os.makedirs(plot_data_dir)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    batch_size = 64
    x_dim  = 2500

    hidden_dim = 64
    latent_dim = 24
    dataset_name = '_Chen_data_9_1_'
    experiment_name = model_type+dataset_name+ str(latent_dim)+"_"+str(hidden_dim)
    
    ## load csv data
    train_rho_data_path = osp.join(lattice_data_dir, 'rho_data_train.csv')
    train_property_data_path = osp.join(lattice_data_dir, 'property_data_train.csv')
    test_rho_data_path = osp.join(lattice_data_dir, 'rho_data_test.csv')
    test_property_data_path = osp.join(lattice_data_dir, 'property_data_test.csv')
    train_X = np.loadtxt(train_rho_data_path, delimiter=',').astype(np.float32)
    ## reshape X to (N, 50, 50) and permute the last two dimensions
    train_X = train_X.reshape(-1, im_x, im_y)
    train_X = np.transpose(train_X, (0, 2, 1))  # Now  X is (N, 50, 50)
    train_y = np.loadtxt(train_property_data_path, delimiter=',').astype(np.float32)  #
    test_X = np.loadtxt(test_rho_data_path, delimiter=',').astype(np.float32)
    test_X = test_X.reshape(-1, im_x, im_y)
    test_X = np.transpose(test_X, (0, 2, 1))  # Now  X is (N, 50, 50)
    test_y = np.loadtxt(test_property_data_path, delimiter=',').astype(np.float32)

    train_dataset = CombinedDataset(train_X, train_y)
    test_X = test_X[:2000,:,:]
    test_y = test_y[:2000,:]
    test_dataset = CombinedDataset(test_X, test_y)

    print("train_dataset shape:{}".format(len(train_dataset)))
    print("test_dataset shape:{}".format(len(test_dataset)))
    ### the min and max of the proeperties
    print("train C11 min: {}, max: {}".format(np.min(train_y[:,0], axis=0), np.max(train_y[:,0], axis=0)))
    print("train C12 min: {}, max: {}".format(np.min(train_y[:,1], axis=0), np.max(train_y[:,1], axis=0)))
    print("train C22 min: {}, max: {}".format(np.min(train_y[:,2], axis=0), np.max(train_y[:,2], axis=0)))
    print("train C33 min: {}, max: {}".format(np.min(train_y[:,3], axis=0), np.max(train_y[:,3], axis=0)))
    print("train rho min: {}, max: {}".format(np.min(train_y[:,4], axis=0), np.max(train_y[:,4], axis=0)))



    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True)
    
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        drop_last = True)

    ### load model
    checkpoints_dir = './checkpoints'
    model_file = osp.join(checkpoints_dir,experiment_name+"_model.pth")
    loaded_model = torch.load(model_file,map_location=torch.device('cpu'))
    loaded_model.device = device
    loaded_model.to(device)
    loaded_model.eval()

    ### check data
    hom_FE_solver = StructuralFE()
    hom_FE_solver.initializeSolver(data_type=torch.float32, nelx=im_x, nely=im_y, penal=2,Emin=1e-6, Emax=1.0, nu=0.3)

    it = iter(train_loader)
    (input, y_true) = next(it)
    input = input.to(device)
    y_true = y_true.float()
    y_true = y_true.to(device)

    print("Original property: {}".format(y_true[0].squeeze()))
    Q_input = hom_FE_solver.solve(input[0].permute(1,0).reshape(im_x*im_y))
    print("Input property:C11:{}, C12:{}, C22:{}, C33:{}, rho:{}".format(Q_input[0,0], Q_input[0,1],Q_input[1,1],Q_input[2,2], torch.mean(input[0])))

    pred, mean, err_e, mean_d, err_d = loaded_model(input)
    print("err_e: {}, err_d:{}".format(torch.mean(err_e), torch.mean(err_d)))

    print("Predicted latent property: {}".format(mean[0,:num_properties].squeeze()))
    pred_density = np.mean(pred[0].cpu().detach().numpy().reshape(im_x,im_y))
    pred[pred<0.5] = 0
    pred[pred>0.5] = 1
    pred = pred.view(-1, im_x, im_y)
    Q = hom_FE_solver.solve(pred[0].permute(1,0).reshape(-1))
    print("Predicted property:C11:{}, C12:{}, C22:{}, C33:{}, rho:{}".format(Q[0,0],Q[0,1],Q[1,1],Q[2,2], pred_density))

    mean2 = mean
    mean2 = mean + 0.01*torch.rand_like(mean).to(device)
    z = loaded_model.Encoder.reparameterization(mean2,err_e)
    pred2, mean_d, err_d2 = loaded_model.Decoder(z)
    print("err_d2:{}".format(torch.mean(err_d2)))

    mean2 = mean
    mean2 = mean + 0.02*torch.rand_like(mean).to(device)
    z = loaded_model.Encoder.reparameterization(mean2,err_e)
    pred2, mean_d, err_d2 = loaded_model.Decoder(z)
    print("err_d3:{}".format(torch.mean(err_d2)))

    mean2 = mean
    mean2 = mean + 0.03*torch.rand_like(mean).to(device)
    z = loaded_model.Encoder.reparameterization(mean2,err_e)
    pred2, mean_d, err_d2 = loaded_model.Decoder(z)
    print("err_d4:{}".format(torch.mean(err_d2)))

    mean2 = mean
    mean2 = mean + 0.04*torch.rand_like(mean).to(device)
    z = loaded_model.Encoder.reparameterization(mean2,err_e)
    pred2, mean_d, err_d2 = loaded_model.Decoder(z)
    print("err_d5:{}".format(torch.mean(err_d2)))

    mean2 = mean
    mean2 = mean + 0.05*torch.rand_like(mean).to(device)
    z = loaded_model.Encoder.reparameterization(mean2,err_e)
    pred2, mean_d, err_d2 = loaded_model.Decoder(z)
    print("err_d5:{}".format(torch.mean(err_d2)))

    ### study the relationship between the spherical error and the distance between a test data from the training dataset
    training_latent_data_path = osp.join(plot_data_dir, experiment_name+'_training_latent_data.txt')
    if osp.exists(training_latent_data_path):
        training_latent_data = np.loadtxt(training_latent_data_path, delimiter=',').astype(np.float32)
    else:
        training_latent_data = []
        for batch_idx, (input, y_true) in enumerate(tqdm(train_loader)):
            input = input.to(device)
            y_true = y_true.float()
            y_true = y_true.to(device)
            _, mean, err_e, _, _ = loaded_model(input)
            training_latent_data.append(mean.cpu().detach().numpy())
        training_latent_data = np.vstack(training_latent_data)
        training_latent_data = np.squeeze(training_latent_data)
        np.savetxt(training_latent_data_path, training_latent_data, delimiter=',', fmt='%.6f')

    test_latent_data = []
    test_sph_err = []
    test_design_err_list = []
    for batch_idx, (input, y_true) in enumerate(tqdm(test_loader)):
        test_x_hat_list = []
        test_target_prop_list = []
        input = input.to(device)
        y_true = y_true.float()
        y_true = y_true.to(device)
        x_hat, mean, err_e, mean_d, err_d = loaded_model(input)

        # x_hat = x_hat.squeeze()
        # test_x_hat_list.append(x_hat.cpu().detach())
        # test_target_prop_list.append(mean[:,:num_properties,0,0].cpu().detach().numpy())
        # test_latent_data.append(mean.cpu().detach().numpy())
        # test_sph_err.append(err_d.cpu().detach().numpy())

        min_noise = 0 
        for _ in range(3):
            # noise = torch.zeros_like(mean,dtype=torch.float32).to(device)
            # noise_indices = torch.randint(0, mean.shape[1], (mean.shape[0],)).to(device)
            # sample_indices = torch.arange(mean.shape[0]).to(device)
            # noise[sample_indices,noise_indices,0,0] = 0.1*torch.rand(mean.shape[0]).to(device)+min_noise
            # mean2 = mean + noise
            mean2 = mean2 + torch.randn_like(mean2)*0.01 + min_noise
            z = loaded_model.Encoder.reparameterization(mean2,err_e)
            pred2, mean_d, err_d2 = loaded_model.Decoder(z)
            test_latent_data.append(mean2.cpu().detach().numpy())
            test_sph_err.append(err_d2.cpu().detach().numpy())
            min_noise += 0.01
            pred2 = pred2.squeeze()
            test_x_hat_list.append(pred2.cpu().detach())
            test_target_prop_list.append(mean2[:,:num_properties,0,0].cpu().detach().numpy())
        test_x_hat_list = torch.cat(test_x_hat_list, dim=0)
        test_target_prop_list = np.concatenate(test_target_prop_list, axis=0)
        sample_range = range(test_x_hat_list.shape[0])
        test_design_err = Parallel(n_jobs=-1)(  # use all available cores
            delayed(compute_error)(idx, test_x_hat_list, test_target_prop_list, hom_FE_solver) for idx in sample_range
        )
        test_design_err = np.array(test_design_err)
        test_design_err_list.append(test_design_err)

    test_latent_data = np.concatenate(test_latent_data, axis=0)
    test_latent_data = np.squeeze(test_latent_data)
    test_sph_err = np.concatenate(test_sph_err, axis=0)  
    test_sph_err = np.squeeze(test_sph_err)
    confidence_score = np.exp(-np.mean(test_sph_err, axis=1))
    test_design_err_list = np.concatenate(test_design_err_list, axis=0)
    print("test_latent_data shape: {}, test_sph_err shape: {}".format(test_latent_data.shape, test_sph_err.shape))
    print("test_design_err shape: {}".format(test_design_err_list.shape))
    min_dist = []
    for i in range(test_latent_data.shape[0]):
        dist = distance.cdist(test_latent_data[i:i+1,:], training_latent_data, 'euclidean')
        min_dist.append(np.min(dist))
    min_dist = np.array(min_dist)
    print("min_dist shape: {}".format(min_dist.shape))
    ### plot the relationship between the confidence score and the min distance
    plt.figure(figsize=(6, 4), dpi=500)
    plt.scatter(min_dist, confidence_score, alpha=0.5)
    plt.xlabel('Min Distance to Training Data in Latent Space')
    plt.ylabel('Confidence Score (exp(-spherical error))')
    plt.grid(True)
    plt.savefig(osp.join(figure_dir, experiment_name+'_confidence_vs_min_dist.png'))
    ### save plot data
    np.savetxt(osp.join(plot_data_dir, experiment_name+'_confidence_vs_min_dist.txt'), np.column_stack((min_dist, confidence_score)), delimiter=',', fmt='%.6f')
    #plt.show()
    # ### plot the relationship between the confidence score and the designed error
    plt.figure(figsize=(6, 4), dpi=500)
    plt.scatter(test_design_err_list, confidence_score, alpha=0.5)
    plt.xlabel('Designed Error')
    plt.ylabel('Confidence Score (exp(-spherical error))')
    plt.grid(True)
    plt.savefig(osp.join(figure_dir, experiment_name+'_confidence_vs_designed_error.png'))
    ### save plot data
    np.savetxt(osp.join(plot_data_dir, experiment_name+'_confidence_vs_designed_error.txt'), np.column_stack((test_design_err_list, confidence_score)), delimiter=',', fmt='%.6f')
