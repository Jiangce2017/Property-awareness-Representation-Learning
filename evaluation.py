import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from models import Model
from utils import loss_function, load_mat, show_image, CombinedDataset, show_image_group

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
    dataset_path = './datasets/Wang/ShapeSpace.mat'
    property_path = './datasets/Wang/PropertySpace.mat'
    results_dir = './results'
    model_file = osp.join("checkpoints","gpu_"+model_type+"_32_64_model.pth")
    batch_size = 64
    x_dim  = 2500
    hidden_dim = 64
    latent_dim = 32
    lr = 1e-3
    epochs = 1000

    ## setup logger
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
        
    model_file = osp.join("checkpoints","well_trained2_"+model_type+"_model.pth")
    loaded_model = torch.load(model_file,map_location=torch.device('cpu'))
    loaded_model.device = device
    loaded_model.to(device)
    loaded_model.eval()

    it = iter(train_loader)
    (input, y_true) = next(it)
    input = input.to(device)
    y_true = y_true.float()
    y_true = y_true.to(device)

    pred, mean, log_var = loaded_model(input)
    loss, reproduction_loss,prediction_loss, var_loss, mean_range = loss_function(input.view(-1,x_dim), pred.view(-1,x_dim),y_true, mean, log_var,model_type,num_properties)
    print("var_loss: {}, mean_range: {}".format(var_loss.item(), mean_range.item()))
    print("reproduction_loss: {}, prediction_loss: {}".format(reproduction_loss.item(), prediction_loss.item()))
    print("loss: {}".format(loss.item()))
    pred = F.sigmoid(pred)
    x_hat_list = []
    titles = []
    x_hat_list.append(input[0].cpu().detach().numpy().reshape(im_x,im_y))
    titles.append("Input Image")
    x_hat_list.append(pred[0].cpu().detach().numpy().reshape(im_x,im_y))   
    titles.append("Predicted Image")

    latent_vector = mean[[0]]

    if model_type == 'Freq_FNO':
        z = loaded_model.reparameterization_FreqNO(mean, log_var)
        print("z[0,0,0] :{}".format(z[0,0,0]))
        torch.manual_seed(123)

        var = 3e-5* torch.ones_like(log_var)
        print("var: {}".format(torch.max(var)))
        epsilon_real = torch.randn(mean.shape[0], latent_dim//2, modes1, modes2).to(device)* torch.sqrt(var[:,:latent_dim//2,:,:]) 
        z_real = mean[:,:latent_dim//2,:,:] + epsilon_real
        epsilon_image = torch.randn(mean.shape[0], latent_dim//2, modes1, modes2).to(device)* torch.sqrt(var[:,latent_dim//2:,:,:])
        z_image = mean[:,latent_dim//2:,:,:] + epsilon_image
        z = torch.complex(z_real, z_image)
        print("z[0,0,0] :{}".format(z[0,0,0]))
        x_hat = loaded_model.Decoder(z,50,50)
        x_hat_list.append(x_hat[0].cpu().detach().numpy().reshape(im_x,im_y))
        titles.append("Reconstructed Image")
    elif model_type == 'FNO':
        latent_vector = torch.tile(latent_vector,(1,1,50,50))
        x_hat = loaded_model.Decoder(latent_vector)
        show_image(x_hat.cpu().detach().numpy().reshape(50,50))
    else:
        x_hat = loaded_model.Decoder(latent_vector)
        show_image(x_hat.cpu().detach().numpy().reshape(50,50))    
    plt.close('all')

    mean_sq = mean.squeeze()
    property_tunning = True
    if property_tunning:
        print("Properties: {}".format(mean_sq[0,:5]))
        mean_sq[0,0] += 0.2
        middle_p = mean_sq[0,:]
        middle_p = middle_p[None,:, None, None]
    else:
        ## print the range of the five proerties
        print("mean shape: {}".format(mean_sq.shape))
        print("min p0: {}, max p0: {}".format(torch.min(mean_sq[:,0]),torch.max(mean_sq[:,0])))
        print("min p1: {}, max p1: {}".format(torch.min(mean_sq[:,1]),torch.max(mean_sq[:,1])))
        print("min p2: {}, max p2: {}".format(torch.min(mean_sq[:,2]),torch.max(mean_sq[:,2])))
        print("min p3: {}, max p3: {}".format(torch.min(mean_sq[:,3]),torch.max(mean_sq[:,3])))
        print("min p4: {}, max p4: {}".format(torch.min(mean_sq[:,4]),torch.max(mean_sq[:,4])))

        ## generate one middle point in inside the simplex
        t = torch.softmax(torch.randn(1, mean_sq.shape[0]),dim=1)
        print("check sigmax t: {}".format(torch.sum(t)))
        middle_p = torch.einsum('ik,kj->ij',t,mean_sq)
        print("middle_p shape: {}".format(middle_p.shape))
        middle_p = middle_p[:,:, None, None]
    var = 3e-5* torch.ones_like(middle_p)
    middle_z = loaded_model.reparameterization_FreqNO(middle_p, var)
    x_hat = loaded_model.Decoder(middle_z)

    x_hat_list.append(x_hat.cpu().detach().numpy().reshape(im_x,im_y))
    titles.append("Property Adjusted Image")
    show_image_group(x_hat_list,titles)





