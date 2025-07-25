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
from utils import loss_function, load_mat, show_image, CombinedDataset,Logger,plot_ternary

if __name__ == '__main__':
    cuda = False
    device = torch.device("cuda" if cuda else "cpu")
    im_x = 50
    im_y = 50 
    num_properties = 5  # Number of properties in the dataset
    modes1 = 10
    modes2 = 6
    model_type = 'FNO'
    dataset_path = './datasets/Wang/ShapeSpace.mat'
    property_path = './datasets/Wang/PropertySpace.mat'
    results_dir = './results'
    model_file = osp.join("checkpoints",model_type+"_model.pth")
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
        
    model_file = osp.join("checkpoints","save_"+model_type+"_model.pth")
    loaded_model = torch.load(model_file)
    loaded_model.eval()
    with torch.no_grad():
        for batch_idx, (input, y_true) in enumerate(tqdm(train_loader)):

            input = input.to(device)
            y_true = y_true.float()
            y_true = y_true.to(device)

            pred, mean, log_var = loaded_model(input)
            loss, *_ = loss_function(input.view(-1,x_dim), pred.view(-1,x_dim),y_true, mean, log_var,model_type,num_properties)
            print("loss: {}".format(loss.item()))
            break
    
    pred = F.sigmoid(pred)
    show_image(input[0].cpu().detach().numpy().reshape(im_x,im_y))
    show_image(pred[0].cpu().detach().numpy().reshape(im_x,im_y))


    latent_vector = mean[[0]]

    
    if model_type == 'Freq_FNO':
        real_latent_vector = latent_vector[:,:latent_dim//2,:,:]
        image_latent_vector = latent_vector[:,latent_dim//2:,:,:]
        latent_vector = torch.complex(real_latent_vector, image_latent_vector)

        latent_vector = torch.tile(latent_vector,(1,1,10,6))
        print("latent_vector shape:{}".format(latent_vector.shape))
        x_hat = loaded_model.Decoder(latent_vector,60,60)
        show_image(x_hat.cpu().detach().numpy().reshape(60,60))
    else:
        latent_vector = torch.tile(latent_vector,(1,1,50,50))
        x_hat = loaded_model.Decoder(latent_vector)
        show_image(x_hat.cpu().detach().numpy().reshape(50,50))
    plt.close('all') 

        # switch to eval mode
    # choose a property vector within your printed ranges:
    desired_props = [0.5, 0.3, 0.7, 0.2, 0.6]
    # generate 4 samples
    samples = loaded_model.generate_by_properties(desired_props, num_samples=4)
    
    # for i, img in enumerate(samples):
    #     show_image(img.squeeze().view(im_x, im_y))
    
    # take the first one, convert to numpy [0,1], then to uint8 [0,255]
    vae_output = samples[0].squeeze().detach().cpu().numpy()
    vae_output = np.clip(vae_output, 0, 1)
    show_image(vae_output.reshape(50,50))
    
    # # save it
    # img_uint8   = (vae_output * 255).astype(np.uint8)
    # Image.fromarray(img_uint8).save("vae_output.png")
    # print(" VAE output saved as vae_output.png")

    print("Reconstruction min/max:", vae_output.min(), vae_output.max())
    
    # # Auto Threshold with Otsu
    # img8 = (vae_output * 255).astype(np.uint8)
    # _, binary = cv2.threshold(img8, 0, 255,
    #                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Image.fromarray(binary).save("vae_output_binary.png")
    # print(" Saved binary lattice as vae_output_binary.png (Otsu threshold)")


            
    # threshold_value = 0.5
    # binary = (vae_output > threshold_value).astype(np.uint8) * 255
    # Image.fromarray(binary).save("vae_output_binary.png")

    # print(" VAE output saved as vae_output.png")

    # display 
    # show_image(torch.tensor(vae_output).view(im_x, im_y))
    
    plt.figure(figsize=(4,4))
    plt.imshow(binary, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()

    




