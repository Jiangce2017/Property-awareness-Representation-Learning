### demonstrate that the proposed method can effectively design lattice structures with desired properties, and a series of lattices
### with varying densities and stiffness

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
import os
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.decomposition import PCA

from models_epi import Model
from utils_epi import loss_function, load_mat, show_image, CombinedDataset, show_image_group, plot_ternary 
from hom_FE import StructuralFE

colors = ["green", "blue"] 
cmap_green_to_blue = LinearSegmentedColormap.from_list("GreenToBlue", colors, N=256)

### set plot text font
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

torch.manual_seed(42)

def get_fixed_value_mask(target_properties, num_initial_vector=5, latent_dim=32):
    fixed_value_mask = torch.zeros((num_initial_vector, latent_dim), dtype=torch.int16).bool()
    for key in target_properties:
        if key == 'C11':
            fixed_value_mask[:,0] = 1
        elif key == 'C12':
            fixed_value_mask[:,1] = 1
        elif key == 'C22':
            fixed_value_mask[:,2] = 1
        elif key == 'C33':
            fixed_value_mask[:,3] = 1
        elif key == 'vf':
            fixed_value_mask[:,4] = 1
    return fixed_value_mask

def get_interpolated_properties(start_properties, end_properties, num_steps):
    property_series = []
    start_properties_array = np.array([start_properties.get(key, 0) for key in start_properties])   
    end_properties_array = np.array([end_properties.get(key, 0) for key in end_properties])
    ### convert numpy to torch
    start_properties = torch.from_numpy(start_properties_array).float()
    end_properties = torch.from_numpy(end_properties_array).float()

    for step in range(num_steps):
        alpha = step / (num_steps - 1)
        target_properties = (1 - alpha) * start_properties + alpha * end_properties
        property_series.append(target_properties)
    property_series = torch.stack(property_series, dim=0)
    return property_series

def property_distance(output_properties, target_properties):
    distance = 0
    for key in target_properties:
        if key == 'C11':
            target_C11 = target_properties[key]
            output_C11 = output_properties[:,0]
            distance += (output_C11 - target_C11)**2/(target_C11+1e-3)**2
        elif key == 'C12':
            target_C12 = target_properties[key]
            output_C12 = output_properties[:,1]
            distance += (output_C12 - target_C12)**2/(target_C12+1e-3)**2
        elif key == 'C22':
            target_C22 = target_properties[key]
            output_C22 = output_properties[:,2]
            distance += (output_C22 - target_C22)**2/(target_C22+1e-3)**2
        elif key == 'C33':
            target_C33 = target_properties[key]
            output_C33 = output_properties[:,3]
            distance += (output_C33 - target_C33)**2/(target_C33+1e-3)**2
        elif key == 'vf':
            target_vf = target_properties[key]
            output_vf = output_properties[:,4]
            distance += (output_vf - target_vf)**2/(target_vf+1e-3)**2
    return distance 

def find_latent_vector(hom_FE_solver, model, target_properties, training_latent_data,latent_dim, im_x, im_y,num_properties, num_initial_vector=5, num_iterations=20, lr=1e-4):
    ### find num_initial_vector vectors in trainning_latent_data that have the closest properties to target_properties
    closest_latent_vectors = []
    closest_distances = []
    for latent_vector in training_latent_data:
        output_properties = latent_vector[None,:num_properties]
        distance = property_distance(output_properties, target_properties)
        if len(closest_distances) < num_initial_vector:
            closest_distances.append(distance)
            closest_latent_vectors.append(latent_vector[None,:])
        else:
            max_distance_index = closest_distances.index(max(closest_distances))
            if distance < closest_distances[max_distance_index]:
                closest_distances[max_distance_index] = distance
                closest_latent_vectors[max_distance_index] = latent_vector[None,:]  

    closest_latent_vectors = np.concatenate(closest_latent_vectors,axis=0)
    closest_latent_vectors = torch.from_numpy(closest_latent_vectors).float().to(model.device)
    ### initialize latent vector with fixed values for the target property, and get the other values from the closest_latent_vector
    latent_vectors = torch.randn((num_initial_vector, latent_dim,1,1), device=model.device)
    latent_vectors[:,:,0,0] = closest_latent_vectors[:,:]
    latent_vectors.requires_grad = True ### change it to a zero tensor

    optimizer = Adam([latent_vectors], lr=lr)

    model.Decoder.x_grid, model.Decoder.y_grid, model.Decoder.dx, model.Decoder.dy = model.Decoder.get_spherical_grid(num_initial_vector, latent_dim, im_x, im_y, device)
    model.eval()
    for iteration in range(num_iterations): 
        optimizer.zero_grad()
        err_e = torch.rand(latent_vectors.shape)*1e-5
        z = loaded_model.Encoder.reparameterization(latent_vectors,err_e)
        selected_x_hat, mean_d, err_d = loaded_model.Decoder(z)
        var_loss = torch.mean(err_d)
        property_loss = torch.mean(property_distance(latent_vectors[:,:num_properties,0,0], target_properties))
        loss = var_loss + property_loss
        loss.backward()
        optimizer.step()
        print(f"Iteration {iteration+1}/{num_iterations}, Var Loss: {var_loss.item()}, Property Loss: {property_loss.item()}, Total Loss: {loss.item()}")

    ### retrieve the final latent vector with the smallest property difference
    selected_x_hat = selected_x_hat.squeeze().detach()
    calculated_properties_list = []
    for i in range(num_initial_vector):
        Q = hom_FE_solver.solve(selected_x_hat[i].permute(1,0).reshape(-1))
        rho = torch.mean(selected_x_hat[i])
        calculated_properties = torch.tensor([Q[0,0], Q[0,1], Q[1,1], Q[2,2], rho], device=model.device)
        calculated_properties_list.append(calculated_properties)
    calculated_properties = torch.stack(calculated_properties_list, dim=0)
    print("Calculated properties shape: {}".format(calculated_properties.shape))
    property_dist = property_distance(calculated_properties, target_properties)
    ### sort property_dist
    sorted_indices = torch.argsort(property_dist)
    property_dist = property_dist[sorted_indices]
    latent_vectors = latent_vectors[sorted_indices,:,:,:].squeeze().detach()
    selected_x_hat = selected_x_hat[sorted_indices,:,:].squeeze().detach()

    return latent_vectors, selected_x_hat, property_dist

def find_latent_vector_fixed_properties(hom_FE_solver, model, target_properties, training_latent_data,latent_dim, im_x, im_y,num_properties, num_initial_vector=5, num_iterations=50, lr=1e-4):
    ### find num_initial_vector vectors in trainning_latent_data that have the closest properties to target_properties
    closest_latent_vectors = []
    closest_distances = []
    for latent_vector in training_latent_data:
        output_properties = latent_vector[None,:num_properties]
        distance = property_distance(output_properties, target_properties)
        if len(closest_distances) < num_initial_vector:
            closest_distances.append(distance)
            closest_latent_vectors.append(latent_vector[None,:])
        else:
            max_distance_index = closest_distances.index(max(closest_distances))
            if distance < closest_distances[max_distance_index]:
                closest_distances[max_distance_index] = distance
                closest_latent_vectors[max_distance_index] = latent_vector[None,:]  

    closest_latent_vectors = np.concatenate(closest_latent_vectors,axis=0)
    closest_latent_vectors = torch.from_numpy(closest_latent_vectors).float().to(model.device)
    closest_latent_vectors = closest_latent_vectors[:,:,None,None]  
    ### initialize latent vector with fixed values for the target property, and get the other values from the closest_latent_vector
    property_mask = get_fixed_value_mask(target_properties, num_initial_vector, latent_dim)
    property_mask = property_mask[:,:,None, None].to(model.device)
    num_free_variables = latent_dim - property_mask[0].sum().item()
    free_variables = torch.zeros((num_initial_vector, num_free_variables,1,1), device=model.device)
    free_variables.requires_grad = True
    optimizer = Adam([free_variables], lr=lr)
    model.Decoder.x_grid, model.Decoder.y_grid, model.Decoder.dx, model.Decoder.dy = model.Decoder.get_spherical_grid(num_initial_vector, latent_dim, im_x, im_y, device)
    model.eval()
    properties_array = np.array([target_properties.get(key, 0) for key in target_properties])  
    properties_array = torch.from_numpy(properties_array).float()
    for iteration in range(num_iterations): 
        optimizer.zero_grad()
        ### assemble free variables and fixed properties into latent_vectors
        latent_vectors = torch.zeros((num_initial_vector, latent_dim,1,1), device=model.device)
        latent_vectors[property_mask] = properties_array.view(1,-1,1,1).repeat(num_initial_vector,1,1,1).view(-1)
        latent_vectors[~property_mask] = closest_latent_vectors[~property_mask] + free_variables.view(-1)
        err_e = torch.rand(latent_vectors.shape)*1e-5
        z = loaded_model.Encoder.reparameterization(latent_vectors,err_e)
        selected_x_hat, mean_d, err_d = loaded_model.Decoder(z)
        loss = torch.mean(err_d)
        loss.backward()
        optimizer.step()
        print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss.item()}")

    ### retrieve the final latent vector with the smallest property difference
    selected_x_hat = selected_x_hat.squeeze().detach()
    calculated_properties_list = []
    for i in range(num_initial_vector):
        Q = hom_FE_solver.solve(selected_x_hat[i].permute(1,0).reshape(-1))
        rho = torch.mean(selected_x_hat[i])
        calculated_properties = torch.tensor([Q[0,0], Q[0,1], Q[1,1], Q[2,2], rho], device=model.device)
        calculated_properties_list.append(calculated_properties)
    calculated_properties = torch.stack(calculated_properties_list, dim=0)
    print("Calculated properties shape: {}".format(calculated_properties.shape))
    property_dist = property_distance(calculated_properties, target_properties)
    ### sort property_dist
    sorted_indices = torch.argsort(property_dist)
    property_dist = property_dist[sorted_indices]
    latent_vectors = latent_vectors[sorted_indices,:,:,:].squeeze().detach()
    selected_x_hat = selected_x_hat[sorted_indices,:,:].squeeze().detach()

    return latent_vectors, selected_x_hat, property_dist

def generate_latent_vector_series(hom_FE_solver, model, start_latent_vector, end_latent_vector, num_steps, latent_dim, im_x, im_y,):
    mode = 'slerp'
    latent_vector_series = []
    if mode == 'simple_interpolation':
        for step in range(num_steps):
            alpha = step / (num_steps - 1)
            target_latent_vector = (1 - alpha) * start_latent_vector + alpha * end_latent_vector
            target_latent_vector = target_latent_vector[None,:,None,None]
            latent_vector_series.append(target_latent_vector)
    latent_vector_series = torch.cat(latent_vector_series, dim=0)
    model.Decoder.x_grid, model.Decoder.y_grid, model.Decoder.dx, model.Decoder.dy = model.Decoder.get_spherical_grid(num_steps, latent_dim, im_x, im_y, model.device)
    with torch.no_grad():
        err_e = torch.rand(latent_vector_series.shape)*1e-5
        z = model.Encoder.reparameterization(latent_vector_series, err_e)
        x_hat_series,mean_d, err_d = model.Decoder(z)
        var_loss = torch.mean(err_d, dim=1)
    latent_vector_series = latent_vector_series.squeeze().detach()
    x_hat_series = x_hat_series.squeeze().detach()
    ### property check
    x_hat_series[x_hat_series<0.5] = 0
    x_hat_series[x_hat_series>0.5] = 1
    for i in range(latent_vector_series.shape[0]):
        Q = hom_FE_solver.solve(x_hat_series[i].permute(1,0).reshape(-1))
        rho = torch.mean(x_hat_series[i])
        calculated_properties = torch.tensor([Q[0,0], Q[0,1], Q[1,1], Q[2,2], rho], device=model.device)
        print("Calculated properties for step {}: {}, var_d: {}".format(i, calculated_properties, var_loss[i].item()))  
    return latent_vector_series, x_hat_series

def optimize_latent_vector_series(hom_FE_solver, model, start_properties, end_properties, num_steps, latent_dim, im_x, im_y, training_latent_data,num_properties):
    ### find the start latent points
    num_initial_vector = 4
    start_latent_vectors, start_x_hat, property_dist = find_latent_vector_fixed_properties(hom_FE_solver,model, start_properties,training_latent_data,latent_dim, im_x, im_y,num_properties, num_initial_vector)
    ### select the latent vectors with property_dist < threshold
    threshold = 0.01
    start_latent_vectors = start_latent_vectors[property_dist < threshold]
    start_x_hat = start_x_hat[property_dist < threshold]
    print("Number of selected start latent vectors: {}".format(start_latent_vectors.shape[0]))
    if start_latent_vectors.shape[0] == 0:
        print("No latent vector found with property distance less than {}".format(threshold))
        return None, None
    else:
        latent_vector_series = []
        x_hat_series = []
        ### interploate properties
        num_selected_start = start_latent_vectors.shape[0]
        property_mask = get_fixed_value_mask(start_properties, num_selected_start, latent_dim)
        property_mask = property_mask[:,:,None, None].to(model.device)
        property_series = get_interpolated_properties(start_properties, end_properties, num_steps)
        delta_l = 1e-2
        num_free_variables = latent_dim - property_mask[0].sum().item()
        lr = 1e-4
        num_iterations = 50
        current_latent_vectors = start_latent_vectors.view(num_selected_start,-1,1,1)
        latent_vector_series.append(current_latent_vectors)
        x_hat_series.append(start_x_hat[None,:,:,:,None])
        print("start_x_hat shape: {}, ".format(start_x_hat.shape))
        for step in range(1, num_steps):
            free_variables = torch.zeros((num_selected_start, num_free_variables,1,1), device=model.device)
            free_variables.requires_grad = True
            optimizer = Adam([free_variables], lr=lr)
            model.Decoder.x_grid, model.Decoder.y_grid, model.Decoder.dx, model.Decoder.dy = model.Decoder.get_spherical_grid(num_selected_start, latent_dim, im_x, im_y, device)
            model.eval()
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                ### assemble free variables and fixed properties into latent_vectors
                latent_vectors = torch.zeros((num_selected_start, latent_dim,1,1), device=model.device)
                latent_vectors[property_mask] = property_series[step].view(1,-1,1,1).repeat(num_selected_start,1,1,1).view(-1)
                latent_vectors[~property_mask] = current_latent_vectors[~property_mask] + free_variables.view(-1)
                err_e = torch.rand(latent_vectors.shape)*1e-5
                z = model.Encoder.reparameterization(latent_vectors,err_e)
                x_hat, mean_d, err_d = model.Decoder(z)
                loss = torch.mean(err_d)
                loss.backward()
                optimizer.step()
                print(f"Step {step+1}/{num_steps}, Iteration {iteration+1}/{num_iterations}, Var Loss: {loss.item()}")
            ### update current_latent_vectors
            current_latent_vectors = latent_vectors.detach()
            latent_vector_series.append(current_latent_vectors)
            x_hat_series.append(x_hat[None,:,:,:,:].detach())
        latent_vector_series = torch.cat(latent_vector_series, dim=1)
        x_hat_series = torch.cat(x_hat_series, dim=0).squeeze()
        print("latent_vector_series shape: {}, x_hat_series shape: {}".format(latent_vector_series.shape, x_hat_series.shape))
        return latent_vector_series, x_hat_series

if __name__ == '__main__':
    cuda = False
    device = torch.device("cuda" if cuda else "cpu")
    im_x = 50
    im_y = 50 
    num_properties = 5  # Number of properties in the dataset
    modes1 = 10
    modes2 = 6
    batch_size = 64
    x_dim  = 2500
    hidden_dim = 64
    latent_dim = 24
    model_type = 'Freq_FNO'
    dataset_name = '_Chen_data_9_1_'

    plot_pca = False
    plot_random_lattice_with_pca = False
    design_lattice_with_property = False
    design_series_lattice = True

    
    results_dir = './results'
    checkpoints_dir = './checkpoints'
    plot_data_dir = './plot_data'
    figure_dir = './figures'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(plot_data_dir):
        os.makedirs(plot_data_dir)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    

    experiment_name = model_type+dataset_name+ str(latent_dim)+"_"+str(hidden_dim)

    model_file = osp.join(checkpoints_dir,experiment_name+"_model.pth")

    
    ## load csv data
    lattice_data_dir = '/scratch/jc14407/datasets/Chen/trainning_test_9_1'
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

    loaded_model = torch.load(model_file,map_location=torch.device('cpu'))
    loaded_model.device = device
    loaded_model.to(device)
    loaded_model.eval()

    hom_FE_solver = StructuralFE()
    hom_FE_solver.initializeSolver(data_type=torch.float32, nelx=im_x, nely=im_y, penal=2,Emin=1e-6, Emax=1.0, nu=0.3)

    ### check model validity
    it = iter(test_loader)
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

    ### import training_latent_data
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
        np.savetxt(osp.join(plot_data_dir, experiment_name+'_training_latent_data.txt'), training_latent_data, delimiter=',', fmt='%.6f')

    ### plot the pca of the training_latent_data
    pca = PCA(n_components=2)
    training_latent_data_pca = pca.fit_transform(training_latent_data)


    #### generate a series of smooth transition lattices given start and end properties
    if design_series_lattice:
        start_properties = {'C11': 0.50, 'C22': 0.30}
        end_properties = {'C11': 0.50, 'C22': 0.70}
        num_steps = 160
        #latent_vector_series, x_hat_series = generate_latent_vector_series(hom_FE_solver, loaded_model, start_latent_vector, end_latent_vector, num_steps, latent_dim, im_x, im_y)
        latent_vector_series, x_hat_series = optimize_latent_vector_series(hom_FE_solver, loaded_model, start_properties, end_properties, num_steps, latent_dim, im_x, im_y,training_latent_data,num_properties)
        lattice_series_figure_dir = osp.join(figure_dir, 'lattice_series')
        if not osp.exists(lattice_series_figure_dir):
            os.makedirs(lattice_series_figure_dir)
        lattice_series_data_dir = osp.join(plot_data_dir, 'lattice_series')
        if not osp.exists(lattice_series_data_dir):
            os.makedirs(lattice_series_data_dir)
        ### save the lattice series data
        data_name = "_series"
        for key, value in start_properties.items():
            data_name += f"_{key}_{value}"
        data_name += "_to"
        for key, value in end_properties.items():
            data_name += f"_{key}_{value}"
        np.savetxt(osp.join(lattice_series_data_dir, experiment_name+data_name+'.txt'), latent_vector_series.squeeze().cpu().detach().numpy(), delimiter=',', fmt='%.6f')
        ### plot the lattice series
        lattice_series_figure_path = osp.join(lattice_series_figure_dir, experiment_name+data_name+'.jpg')
        num_series = x_hat_series.shape[1]
        plotted_lattice_series_indice = np.arange(0, num_steps, max(1,num_steps//8))
        plotted_lattice_series_number = len(plotted_lattice_series_indice)
        x_hat_series = x_hat_series[plotted_lattice_series_indice,:,:,:]
        fig, axs = plt.subplots(num_series, plotted_lattice_series_number, figsize=(4*plotted_lattice_series_number, 4*num_series))
        for j in range(num_series):
            for i in range(plotted_lattice_series_number):
                axs[j, i].imshow(x_hat_series[i][j].cpu().detach().numpy(), cmap='Greens')
                ### calculate the properties with hom_FE_solver
                Q = hom_FE_solver.solve(x_hat_series[i][j].permute(1,0).reshape(-1))
                rho = torch.mean(x_hat_series[i][j])
                axs[j, i].set_title(f'Step {i}\nC11:{Q[0,0]:.3f}, C22:{Q[1,1]:.3f}, rho:{rho:.2f}')
                axs[j, i].axis('off')

        plt.savefig(lattice_series_figure_path)
        plt.close(fig)

    if design_lattice_with_property:
        ### given specific properties, generate the lattices satisfying those properties.
        num_initial_vector = 16
        target_C11 = 0.30
        target_C12 = 0.10
        target_C22 = 0.40
        target_C33 = 0.20
        target_vf = 0.40
        target_properties = {'C11': target_C11, 'C22': target_C22}
        result_name = "_"
        for key, value in target_properties.items():
            result_name += f"{key}_{value}"
        data_file_path = osp.join(plot_data_dir, experiment_name+result_name+'.txt')
        x_hat_file_path = osp.join(plot_data_dir, experiment_name+result_name+'_x_hat.txt')
        if False and osp.exists(data_file_path) and osp.exists(x_hat_file_path):
            ### load selected_latent_vectors
            selected_latent_vectors = np.loadtxt(data_file_path, delimiter=',').astype(np.float32)
            selected_latent_vectors = torch.from_numpy(selected_latent_vectors).float().to(device)
            selected_latent_vectors = selected_latent_vectors.view(-1, latent_dim)
            ### load selected_x_hat
            selected_x_hat = np.loadtxt(x_hat_file_path, delimiter=',').astype(np.float32)
            selected_x_hat = torch.from_numpy(selected_x_hat).float().to(device)
            selected_x_hat = selected_x_hat.view(-1, im_x, im_y).permute(0,2,1)

        ### find the latent vector that satisfies the target properties
        else:
            selected_latent_vectors, selected_x_hat, property_dist = find_latent_vector_fixed_properties(hom_FE_solver,loaded_model, target_properties,training_latent_data,latent_dim, im_x, im_y,num_properties, num_initial_vector)
            ### save selected_latent_vectors
            np.savetxt(data_file_path, selected_latent_vectors.cpu().detach().numpy(), delimiter=',', fmt='%.6f')
            ### save selected_latent_x_hat
            np.savetxt(x_hat_file_path, selected_x_hat.permute(0,2,1).reshape(-1, im_x*im_y).cpu().detach().numpy(), delimiter=',', fmt='%.6f')
        ### print the first 5 lattices
        num_initial_vector = 16
        num_show_vector = 5
        print("Target Properties:")
        for key, value in target_properties.items():
            print(f"  {key}: {value}")
        print("Designed Properties:")
        fixed_value_mask = get_fixed_value_mask(target_properties, num_initial_vector, latent_dim)
        print(selected_latent_vectors[fixed_value_mask].cpu().detach().numpy())
        ### evaluate the results with hom_FEM
        selected_x_hat[selected_x_hat<0.5] = 0
        selected_x_hat[selected_x_hat>=0.5] = 1
        print("Homogenized Results:")
        for i in range(num_show_vector):
            Q = hom_FE_solver.solve(selected_x_hat[i].permute(1,0).reshape(-1))
            print(f"  Sample {i}: {Q}")
        ### plot selected_x_hat
        fig, axs = plt.subplots(1, num_show_vector, figsize=(4*num_show_vector, 4))
        for i in range(num_show_vector):
            axs[i].imshow(selected_x_hat[i].cpu().detach().numpy(), cmap='Greens')
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(osp.join(figure_dir, experiment_name+result_name+'.jpg'))
        plt.close()

    ### plot lattices with pca
    ### generate n random indices from the training data
    if plot_random_lattice_with_pca:
        n = 9
        np.random.seed(123456)
        random_indices = np.random.choice(training_latent_data.shape[0], size=n, replace=False)
        selected_latent_vectors = training_latent_data[random_indices]
        err_e = torch.rand(n, selected_latent_vectors.shape[1],1,1)*1e-5
        selected_latent_vectors = torch.from_numpy(selected_latent_vectors).float().to(device)
        selected_latent_vectors = selected_latent_vectors[:,:,None, None]
        loaded_model.Decoder.x_grid, loaded_model.Decoder.y_grid, loaded_model.Decoder.dx, loaded_model.Decoder.dy = loaded_model.Decoder.get_spherical_grid(n, latent_dim, im_x, im_y, device)
        z = loaded_model.Encoder.reparameterization(selected_latent_vectors,err_e)
        selected_x_hat, mean_d, err_d = loaded_model.Decoder(z)
        selected_x_hat[selected_x_hat>0.5] = 1
        selected_x_hat[selected_x_hat<=0.5] = 0
        ### plot selected_x_hat
        fig, axs = plt.subplots(3, 3, figsize=(12, 9))
        for i in range(9):
            axs[i // 3, i % 3].imshow(selected_x_hat[i].cpu().detach().numpy(), cmap='Greens')
            axs[i // 3, i % 3].set_title(f'{i}')
            ## set title font size
            axs[i // 3, i % 3].title.set_fontsize(72)
            axs[i // 3, i % 3].axis('off')
        plt.tight_layout()
        plt.savefig(osp.join(figure_dir, experiment_name+'_selected_x_hat.jpg'))
        plt.close()

        ### plot the selected x_hat with pca projection
        selected_x_hat_pca = training_latent_data_pca[random_indices[:9]]
        plt.figure(figsize=(6, 4), dpi=500)
        
        c = (training_latent_data[:, 0] - training_latent_data[:, 0].min()) / (training_latent_data[:, 0].max() - training_latent_data[:, 0].min())
        norm = matplotlib.colors.Normalize(vmin=training_latent_data[:, 0].min(), vmax=training_latent_data[:, 0].max())
        plt.scatter(training_latent_data_pca[:, 0], training_latent_data_pca[:, 1], c=c, s=5, norm=norm, cmap='viridis')
        plt.colorbar(label='$C_{11}$')
        plt.scatter(selected_x_hat_pca[:, 0], selected_x_hat_pca[:, 1], s=10, color='black')
        ### show the number in form of text near each selected_x_hat_pca
        for i in range(selected_x_hat_pca.shape[0]):
            plt.text(selected_x_hat_pca[i, 0], selected_x_hat_pca[i, 1], str(i), fontsize=16, color='black')
        plt.title('PCA of Latent Vectors Colored by $C_{11}$')
        plt.xlabel('$PCA_1$')
        plt.ylabel('$PCA_2$')
        plt.axis('equal')
        plt.savefig(osp.join(figure_dir, experiment_name+'_selected_x_hat_pca.jpg'))
        plt.close()

    if plot_pca:
        ### plot in a high resolution
        plt.figure(figsize=(6, 4), dpi=500)
        plt.scatter(training_latent_data_pca[:, 0], training_latent_data_pca[:, 1], s=5)
        #plt.title('PCA of Latent Vectors')
        plt.xlabel('$PCA_1$')
        plt.ylabel('$PCA_2$')
        ### set equal aspect ratio
        plt.axis('equal')
        plt.savefig(osp.join(figure_dir, experiment_name+'_training_latent_data_pca.jpg'))
        plt.close()

        ### plot the pca of the traning_latent_data with colors according to the first component of the latent vector which is C11 property
        plt.figure(figsize=(6, 4), dpi=500)
        ### normalize the color values,
        c = (training_latent_data[:, 0] - training_latent_data[:, 0].min()) / (training_latent_data[:, 0].max() - training_latent_data[:, 0].min())
        norm = matplotlib.colors.Normalize(vmin=training_latent_data[:, 0].min(), vmax=training_latent_data[:, 0].max())
        plt.scatter(training_latent_data_pca[:, 0], training_latent_data_pca[:, 1], c=c, s=5, norm=norm, cmap='viridis')
        #plt.title('PCA of Latent Vectors Colored by $C_{11}$')
        ### add a subscrip in the xlabel
        plt.xlabel('$PCA_1$')
        plt.ylabel('$PCA_2$')
        plt.axis('equal')
        plt.colorbar(label='$C_{11}$')
        plt.savefig(osp.join(figure_dir, experiment_name+'_training_latent_data_pca_c11.jpg'))
        plt.close()

        ### plot the pca of the traning_latent_data with colors according to the second component of the latent vector which is C12 property
        plt.figure(figsize=(6, 4), dpi=500)
        ### normalize the color values,
        c = (training_latent_data[:, 1] - training_latent_data[:, 1].min()) / (training_latent_data[:, 1].max() - training_latent_data[:, 1].min())
        norm = matplotlib.colors.Normalize(vmin=training_latent_data[:, 1].min(), vmax=training_latent_data[:, 1].max())
        plt.scatter(training_latent_data_pca[:, 0], training_latent_data_pca[:, 1], c=c, s=5, norm=norm, cmap='viridis')
        #plt.title('PCA of Latent Vectors Colored by $C_{12}$')
        plt.xlabel('$PCA_1$')
        plt.ylabel('$PCA_2$')
        plt.axis('equal')
        plt.colorbar(label='$C_{12}$')
        plt.savefig(osp.join(figure_dir, experiment_name+'_training_latent_data_pca_c12.jpg'))
        plt.close()

        ### plot the pca of the training_latent_data with colors according to the third component of the latent vecotr, which is C22 property
        plt.figure(figsize=(6, 4), dpi=500)
        ### normalize the color values,
        c = (training_latent_data[:, 2] - training_latent_data[:, 2].min()) / (training_latent_data[:, 2].max() - training_latent_data[:, 2].min())
        norm = matplotlib.colors.Normalize(vmin=training_latent_data[:, 2].min(), vmax=training_latent_data[:, 2].max())
        plt.scatter(training_latent_data_pca[:, 0], training_latent_data_pca[:, 1], c=c, s=5, norm=norm, cmap='viridis')
        #plt.title('PCA of Latent Vectors Colored by $C_{22}$')
        plt.xlabel('$PCA_1$')
        plt.ylabel('$PCA_2$')
        plt.axis('equal')
        plt.colorbar(label='$C_{22}$')
        plt.savefig(osp.join(figure_dir, experiment_name+'_training_latent_data_pca_c22.jpg'))
        plt.close()

        ### plot the pca of the training_latent_data with colors according to the forth component of the latent vector, which is C33 property
        plt.figure(figsize=(6, 4), dpi=500)
        ### normalize the color values,
        c = (training_latent_data[:, 3] - training_latent_data[:, 3].min()) / (training_latent_data[:, 3].max() - training_latent_data[:, 3].min())
        norm = matplotlib.colors.Normalize(vmin=training_latent_data[:, 3].min(), vmax=training_latent_data[:, 3].max())
        plt.scatter(training_latent_data_pca[:, 0], training_latent_data_pca[:, 1], c=c, s=5, norm=norm, cmap='viridis')
        #plt.title('PCA of Latent Vectors Colored by $C_{33}$')
        plt.xlabel('$PCA_1$')
        plt.ylabel('$PCA_2$')
        plt.axis('equal')
        plt.colorbar(label='$C_{33}$')
        plt.savefig(osp.join(figure_dir, experiment_name+'_training_latent_data_pca_c33.jpg'))
        plt.close()

        ### plot the pca of the training_latent_data with colors according to the fifth component of the latent vector, which is density
        plt.figure(figsize=(6, 4), dpi=500)
        ### normalize the color values,
        c = (training_latent_data[:, 4] - training_latent_data[:, 4].min()) / (training_latent_data[:, 4].max() - training_latent_data[:, 4].min())
        norm = matplotlib.colors.Normalize(vmin=training_latent_data[:, 4].min(), vmax=training_latent_data[:, 4].max())
        plt.scatter(training_latent_data_pca[:, 0], training_latent_data_pca[:, 1], c=c, s=5, norm=norm, cmap='viridis')

        #plt.title('PCA of Latent Vectors Colored by $v_f$')
        plt.xlabel('$PCA_1$')
        plt.ylabel('$PCA_2$')
        plt.axis('equal')
        plt.colorbar(label='$v_f$')
        plt.savefig(osp.join(figure_dir, experiment_name+'_training_latent_data_pca_density.jpg'))
        plt.close()
