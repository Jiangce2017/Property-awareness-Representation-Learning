# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import os.path as osp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from models import Model
from utils import CombinedDataset,loss_function, load_mat, show_image, train_model, test_model,Logger




warmup_epochs = 10         # number of epochs to ramp KL-weight up
kl_max_weight = 1.0        # final weight of KL term


if __name__ == '__main__':
    cuda = False
    device = torch.device("cuda" if cuda else "cpu")
    im_x = 50
    im_y = 50 
    model_type = 'FNO'
    dataset_path = './datasets/Wang/ShapeSpace.mat'
    property_path = './datasets/Wang/PropertySpace.mat'
    batch_size = 16
    x_dim  = 2500
    hidden_dim = 64
    latent_dim = 32
    num_properties = 5  # Number of properties in the dataset
    lr = 1e-4
    epochs = 80
    results_dir = './results'
    model_file = osp.join("checkpoints",model_type+"_model.pth")
    ## setup logger
    train_logger = Logger(
        osp.join(results_dir, model_type+'_train.log'),
        ['ep', 'train_loss','train_rep','train_pred','train_var','train_mean']
    )
    test_logger = Logger(
        osp.join(results_dir, model_type+'_test.log'),
        ['ep', 'test_loss','test_rep', 'test_pred','test_var','test_mean']
    )

    kwargs = {'num_workers': 1, 'pin_memory': False} 


    mat_data = load_mat(dataset_path)       #ShapeSpace
    prop_data = load_mat(property_path)     #PropertySpace
    
    X = mat_data['ShapeSpace'].astype(np.float32)
    y = prop_data['PropertySpace'].astype(np.float32).transpose()   
    print("X shape:{}, y shape: {}".format(X.shape, y.shape))
    
    # Compute min/max per property
    prop_min = np.min(y, axis=0)
    prop_max = np.max(y, axis=0)
    print("Property ranges:")
    for i, (lo, hi) in enumerate(zip(prop_min, prop_max), 1):
        print(f"  Property {i}:  min = {lo:.4f},  max = {hi:.4f}")

    
    dataset = CombinedDataset(X,y)

    print("dataset shape:{}".format(len(dataset)))
    
    from torch.utils.data import Subset
    
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
    
    # ── ADD THESE LINES FOR A SMALL DEBUG SUBSET 
    debug_n = 200
    train_dataset = Subset(train_dataset, list(range(debug_n)))
    print(f"  Debug mode: training on only {debug_n} samples "  f"→ {len(train_dataset)/batch_size:.0f} batches/epoch")

    debug_n = 20
    test_dataset = Subset(test_dataset, list(range(debug_n)))
    print(f"  Debug mode: testing on only {debug_n} samples "  f"→ {len(test_dataset)/batch_size:.0f} batches/epoch")


    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        **kwargs )
    
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False,
        **kwargs
        )
    
    print("ShapeSpace min:", X.min(), "max:", X.max())
    print("PropertySpace min:", y.min(), "max:", y.max())
    
    print("train_loader shape: {}".format(len(train_loader)))

    model = Model(num_properties,x_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y, modes1=10, modes2=6).to(device)
    print("Model created with type:", model_type)
    optimizer = Adam(model.parameters(), lr=lr)
    # New Scheduler to reduce loss 
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',        # we want to minimize the loss
        factor=0.5,        # multiply LR by 0.5 whenever we trigger
        patience=2,        # wait 2 epochs without improvement
        )

    print("Start training VAE...")
    print(f" Training for {epochs} epochs with batch size {batch_size}…")
    for epoch in range(epochs):
        overall_loss, rep_loss, pred_loss, m_loss, v_loss = train_model(train_loader,model,device,optimizer,x_dim,model_type,num_properties)
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Train Loss: ", overall_loss)
        train_logger.log({
        'ep': epoch,             
        'train_loss': overall_loss,
        'train_rep': rep_loss,
        'train_pred': pred_loss,
        'train_var': v_loss, 
        'train_mean': m_loss
        })
        if epoch % 10 == 0:
            torch.save(model, model_file)
            overall_loss, rep_loss, pred_loss, m_loss, v_loss = test_model(test_loader, model,device,x_dim,model_type,num_properties)
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Test Loss: ", overall_loss)
            test_logger.log({
            'ep': epoch,             
            'test_loss': overall_loss,
            'test_rep': rep_loss,
            'test_pred': pred_loss,
            'test_var': v_loss,
            'test_mean': m_loss
            })
        
    print("Finish!!")
    torch.save(model, model_file)



