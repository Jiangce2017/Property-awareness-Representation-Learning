import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from models import PropertyPredictor
from utils import CombinedDataset,loss_function, load_mat


if __name__ == '__main__':
    cuda = False
    device = torch.device("cuda" if cuda else "cpu")
    train_model = False
    im_x = 50
    im_y = 50

    shape_dataset_path = './datasets/Wang/ShapeSpace.mat'
    property_dataset_path = './datasets/Wang/PropertySpace.mat'
    batch_size = 32
    input_dim  = 2500
    hidden_dim = 64
    output_dim = 5
    lr = 1e-3
    epochs = 10

    kwargs = {'num_workers': 1, 'pin_memory': False}

    shape_data = load_mat(shape_dataset_path)
    input_dataset = shape_data['ShapeSpace'].astype(np.float32)

    property_data = load_mat(property_dataset_path)
    output_dataset = property_data['PropertySpace'].astype(np.float32).transpose()

    combined_dataset = CombinedDataset(input_dataset, output_dataset)

    train_dataset, test_dataset = train_test_split(combined_dataset, test_size=0.1, random_state=42)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False,drop_last=True, **kwargs)
        
    predictor = PropertyPredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

    if train_model:
        optimizer = Adam(predictor.parameters(), lr=lr)
        print("Start training predictor...")
        predictor.train()
        for epoch in range(epochs):
            overall_loss = 0
            for batch_idx, (x, y_true) in enumerate(train_loader):

                x = x.to(device)
                y_true = y_true.to(device)

                optimizer.zero_grad()

                y_pred = predictor(x)
                loss = loss_function(y_pred,y_true)
                
                overall_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
            
        print("Finish!!")
        torch.save(predictor, './checkpoints/metalattice_model.pth')
    
        