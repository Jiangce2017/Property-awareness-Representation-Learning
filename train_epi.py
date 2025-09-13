# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
import os.path as osp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from models_epi import Model
from utils_epi import CombinedDataset,loss_function, load_mat, show_image, train_model, test_model,Logger

if __name__ == '__main__':
    cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    im_x = 50
    im_y = 50 
    batch_size = 64
    x_dim  = 2500
    hidden_dim = 64
    latent_dim = 24
    num_properties = 5  # Number of properties in the dataset
    model_type = 'Freq_FNO'  # Options: 'FNO', 'Freq_FNO', 'CNN', 'FL', 'Spherical_FNO'
    dataset_root_dir = '/scratch/jc14407/datasets'

    split_ratio = 0.9
    lattice_data_dir = '/scratch/jc14407/datasets/Chen/trainning_test_9_1'

    lr = 1e-4
    epochs = 5000
    modes1 = 10
    modes2 = 6

    experiment_name = model_type+"_Chen_data_9_1_"+ str(latent_dim)+"_"+str(hidden_dim)
    use_old_model = True

    train_rho_data_path = osp.join(lattice_data_dir, 'rho_data_train.csv')
    train_property_data_path = osp.join(lattice_data_dir, 'property_data_train.csv')
    test_rho_data_path = osp.join(lattice_data_dir, 'rho_data_test.csv')
    test_property_data_path = osp.join(lattice_data_dir, 'property_data_test.csv')
    if not osp.exists(train_rho_data_path):
        print("Dataset not found at", train_rho_data_path)
        exit(1)
    if not osp.exists(train_property_data_path):
        print("Property dataset not found at", train_property_data_path)
        exit(1)
    if not osp.exists(test_rho_data_path):
        print("Dataset not found at", test_rho_data_path)
        exit(1)
    if not osp.exists(test_property_data_path):
        print("Property dataset not found at", test_property_data_path)
        exit(1)
    checkpoints_dir = './checkpoints'
    if not osp.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
    model_file = osp.join(checkpoints_dir,experiment_name+"_model.pth")    
    
    results_dir = './results'
    if not osp.exists(results_dir):
        os.makedirs(results_dir)
    
    ## setup logger
    train_logger = Logger(
        osp.join(results_dir, experiment_name+'_train.log'),
        ['ep', 'train_loss','train_rep','train_pred','train_sph_err_e','train_sph_err_d','train_mean']
    )
    test_logger = Logger(
        osp.join(results_dir, experiment_name+'_test.log'),
        ['ep', 'test_loss','test_rep', 'test_pred','test_sph_err_e','test_sph_err_d','test_mean']
    )

    ## load csv data
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
    test_dataset = CombinedDataset(test_X, test_y)

    print("train_dataset shape:{}".format(len(train_dataset)))
    print("test_dataset shape:{}".format(len(test_dataset)))

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

    if osp.exists(model_file) and use_old_model:
        print("Loading model from", model_file)
        model = torch.load(model_file,map_location=device)
        model.device = device
    else:
        model = Model(batch_size,num_properties,x_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y, modes1=10, modes2=6).to(device)

    print("Model created with type:", model_type)
    optimizer = Adam(model.parameters(), lr=lr)
    # New Scheduler to reduce loss 
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',        # we want to minimize the loss
        factor=0.5,        # multiply LR by 0.5 whenever we trigger
        patience=2,        # wait 2 epochs without improvement
        )

    print("Start training model...")
    print(f" Training for {epochs} epochs with batch size {batch_size}…")
    for epoch in range(epochs):
        overall_loss, rep_loss, pred_loss, sph_err_e, sph_err_d, m_loss = train_model(train_loader,model,device,optimizer,x_dim,model_type,num_properties)
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Train Loss: ", overall_loss)
        train_logger.log({
        'ep': epoch,             
        'train_loss': overall_loss,
        'train_rep': rep_loss,
        'train_pred': pred_loss,
        'train_sph_err_e': sph_err_e,
        'train_sph_err_d': sph_err_d,
        'train_mean': m_loss
        })
        if epoch % 10 == 0:
            torch.save(model, model_file)
            overall_loss, rep_loss, pred_loss, sph_err_e, sph_err_d, m_loss = test_model(test_loader, model,device,x_dim,model_type,num_properties)
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Test Loss: ", overall_loss)
            test_logger.log({
            'ep': epoch,             
            'test_loss': overall_loss,
            'test_rep': rep_loss,
            'test_pred': pred_loss,
            'test_sph_err_e': sph_err_e,
            'test_sph_err_d': sph_err_d,
            'test_mean': m_loss
            })
        
    print("Finish!!")
    torch.save(model, model_file)



