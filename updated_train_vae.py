# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

import torch.nn.utils as nn_utils

from utils_FNO import loss_function as fno_loss

from sklearn.model_selection import train_test_split
from updated_models_vae import Model
from updated_utils import loss_function_vae, load_mat, show_image
from utils import CombinedDataset 


property_path = './datasets/Wang/PropertySpace.mat'

warmup_epochs = 10         # number of epochs to ramp KL-weight up
kl_max_weight = 1.0        # final weight of KL term


if __name__ == '__main__':

    """
        A simple implementation of Gaussian MLP Encoder and Decoder
    """
    cuda = False
    device = torch.device("cuda" if cuda else "cpu")
    train_model = True
    im_x = 50
    im_y = 50 
    model_type = 'FNO'
    dataset_path = './datasets/Wang/ShapeSpace.mat'
    batch_size = 500
    x_dim  = 2500
    hidden_dim = 64
    latent_dim = 32
    lr = 1e-4
    epochs = 20
    mnist_transform = transforms.Compose([
            transforms.ToTensor(),
    ])

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
    debug_n = 2000
    train_dataset = Subset(train_dataset, list(range(debug_n)))
    print(f"  Debug mode: training on only {debug_n} samples "  f"→ {len(train_dataset)/batch_size:.0f} batches/epoch")

    
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

    model = Model(x_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y, modes1=10, modes2=6).to(device)

    if train_model:
        optimizer = Adam(model.parameters(), lr=lr)
        
        # New Scheduler to reduce loss 
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',        # we want to minimize the loss
            factor=0.5,        # multiply LR by 0.5 whenever we trigger
            patience=2,        # wait 2 epochs without improvement
            )

        print("Start training VAE...")
        model.train()
        print(f" Training for {epochs} epochs with batch size {batch_size}…")
        
        for epoch in range(epochs):
            
            overall_loss = 0
            
            # start of epoch
            print(f"\n─── Starting epoch {epoch+1}/{epochs} ───")
            
            for batch_idx, (x, y) in enumerate(train_loader):
                    
                #print("batch traning")
                x = x.view(batch_size, x_dim)
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                x_hat, material_pred, log_var = model(x)
                
                
                x_flat = x.view(x.size(0), -1)
                x_hat_flat = x_hat.view(x_hat.size(0), -1)
                
                # compute losses
                x_flat     = x.view(x.size(0), -1)
                x_hat_flat = x_hat.view(x_hat.size(0), -1)

                #print("x_flat min/max:", x_flat.min().item(), x_flat.max().item())
                #print("x_hat_flat min/max:", x_hat_flat.min().item(), x_hat_flat.max().item())
                #print("material_pred shape:", material_pred.shape)
                #print("y shape:", y.shape)
                #print("material_pred min/max:", material_pred.min().item(), material_pred.max().item())
                #print("y min/max:", y.min().item(), y.max().item())

                kl_weight = min(kl_max_weight, (epoch + 1) / warmup_epochs)
                
                if model_type in ('FNO', 'Freq_FNO'):
                    # fno_loss should return at least the total loss as its first output
                    loss, *_ = fno_loss(
                        x.view(-1, x_dim),           # true flattened
                        x_hat.view(-1, x_dim),       # pred flattened
                        material_pred,               # here is your “mean”
                        log_var,                     # here is your “log_var”
                        model_type                   # if your loss needs to know real vs complex
                        )
                else: 
                    loss = loss_function_vae(x_flat, x_hat_flat, material_pred, y, log_var, kl_weight)
                
                #overall_loss += loss.item()
                
                loss.backward()
                nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                overall_loss += loss.item()
                
            avg_loss = overall_loss / len(train_loader)
            print(f" Epoch {epoch+1}/{epochs} complete — avg loss = {avg_loss:.4f}")
            
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(overall_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"    reducing learning rate to {new_lr:.2e}")
            else:
                print(f"    learning rate remains {new_lr:.2e}")            
            
        print("Finish!!")
        torch.save(model, './checkpoints/metalattice_model.pth')
        
        # switch to eval mode
        model.eval()
        # choose a property vector within your printed ranges:
        desired_props = [0.5, 0.3, 0.7, 0.2, 0.6]
        # generate 4 samples
        samples = model.generate_by_properties(desired_props, num_samples=4)
        # display them
        for i, img in enumerate(samples):
            show_image(img.squeeze().view(im_x, im_y))


    else:
        loaded_model = torch.load('./checkpoints/metalattice_model.pth')
        loaded_model.eval()
        with torch.no_grad():
            for batch_idx, x in enumerate(tqdm(test_loader)):
                print(x.shape)
                x = x.view(batch_size, x_dim)
                x = x.to(device)
                
                x_hat, _, _ = loaded_model(x)
                break
        
        show_image(x[0])
        show_image(x_hat[0].view(im_x,im_y))

        with torch.no_grad():
            noise = torch.randn(batch_size, latent_dim).to(device)
            generated_images = loaded_model.Decoder(noise)
        
        show_image(generated_images[0].view(im_x,im_y))





