import torch
import torch.nn as nn
import numpy as np

class PropertyPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(PropertyPredictor, self).__init__()
            self.FC1 = nn.Linear(input_dim, hidden_dim)
            self.FC2 = nn.Linear(hidden_dim, hidden_dim)
            self.FC3  = nn.Linear(hidden_dim, output_dim)
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.input_dim = input_dim
            
        def forward(self, x):
            x = x.view(-1,self.input_dim)
            h_       = self.LeakyReLU(self.FC1(x))
            h_       = self.LeakyReLU(self.FC2(h_))
            output     = self.FC3(h_)                                               
            return output
        
    