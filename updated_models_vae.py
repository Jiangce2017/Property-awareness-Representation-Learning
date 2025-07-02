# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,x_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y):
        super(Model, self).__init__()
        self.device = device
        self.num_properties = 5
        
        if model_type == 'CNN':
            self.Encoder = CNN_Encoder(input_dim=1, hidden_dim=hidden_dim, latent_dim=latent_dim,im_x=im_x, im_y=im_y, num_properties=self.num_properties)
            self.Decoder = CNN_Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim,im_x=im_x, im_y=im_y)
        elif model_type == 'FL':
            self.Encoder = FL_Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
            self.Decoder = FL_Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
    def forward(self, x):
        material_pred, latent_z, log_var = self.Encoder(x)
        
        z = self.reparameterization(latent_z, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        z_combined = torch.cat([material_pred, z], dim=1)
        x_hat = self.Decoder(z_combined)

        return x_hat, material_pred, log_var    

class FL_Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super(FL_Encoder, self).__init__()
            self.FC_input = nn.Linear(input_dim, hidden_dim)
            self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
            self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
            self.FC_var   = nn.Linear (hidden_dim, latent_dim)
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.training = True
            self.input_dim = input_dim
            
        def forward(self, x):
            x = x.view(-1, self.input_dim)
            h_       = self.LeakyReLU(self.FC_input(x))
            h_       = self.LeakyReLU(self.FC_input2(h_))
            mean     = self.FC_mean(h_)
            log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                        #             (i.e., parateters of simple tractable normal distribution "q"
            return mean, log_var
        
class FL_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(FL_Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    

class CNN_Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim, im_x, im_y, num_properties):
            
            super(CNN_Encoder, self).__init__()
            
            self.num_properties = num_properties
            
            # 1 input channel --> hidden_dim channels 
            self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(3, 3), stride=1, padding=1)
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=1, padding=1)
            self.maxpool = nn.MaxPool2d(kernel_size=(2, 2)) ## half spatial dimension 
            
            self.conv3 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=(3, 3), stride=1, padding=1)
            self.conv4 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=(3, 3), stride=1, padding=1)
            self.conv5 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=(3, 3), stride=1, padding=1)
            
            self.flatten = nn.Flatten()
            # self.dense1 = nn.Linear(im_x*im_y*16, hidden_dim)
            self.dense1 = nn.Linear((hidden_dim * 2) * 25 * 25, hidden_dim)
            
            
            self.layer_mean = nn.Linear(hidden_dim, latent_dim)
            self.layer_variance = nn.Linear(hidden_dim, latent_dim)
            
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.im_x = im_x
            self.im_y = im_y
            
        def forward(self, x):
            x = x.view(-1,1,self.im_x,self.im_y)  # reshape to (batch_size, channels, height, width)
            
            # complete the forward function
            x = self.LeakyReLU(self.conv1(x))
            x = self.LeakyReLU(self.conv2(x))
            x = self.maxpool(x)  # reduces size by 2
            
            x = self.LeakyReLU(self.conv3(x))
            x = self.LeakyReLU(self.conv4(x))
            x = self.LeakyReLU(self.conv5(x))


            x = self.flatten(x)  # flatten the tensor
            x = self.LeakyReLU(self.dense1(x))  # fully connected layer
            
            mean = self.layer_mean(x)       # one output: mean
            log_var = self.layer_variance(x)  # one output: log(variance)
            
            material_pred = mean[:, :self.num_properties]
            latent_z = mean[:, self.num_properties:]
            
            return material_pred, latent_z, log_var
    
    
class CNN_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim,im_x,im_y):
        super(CNN_Decoder, self).__init__()
        
        self.init_channels = hidden_dim // 4                   # start decoding from fewer channels 
        
        
        self.dense1 = nn.Linear(latent_dim, im_x*im_y*2)
        self.dense2 = nn.Linear(im_x*im_y*2, self.init_channels * 25 * 25)
        # self.dense2 = nn.Linear(im_x*im_y*2,im_x*im_y*16)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')  # from 25x25 --> 50x50
        
        
        # self.conv1 = nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv1 = nn.Conv2d(self.init_channels, hidden_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=(3, 3), stride=1, padding=1)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()   # to keep BCE-compatible output [0,1]
        
        self.hidden_dim = hidden_dim
        self.im_x = im_x
        self.im_y = im_y

        
    def forward(self, x):
        # complete the forward function
        x = self.LeakyReLU(self.dense1(x))  # Expand latent vector
        x = self.LeakyReLU(self.dense2(x))  # Expand more

        # Reshape to 4D tensor (batch_size, channels, height, width)
       
        x = x.view(x.size(0), self.init_channels, 25, 25)

        # Upsample back to original image size
        x = self.upsample(x)  # Doubles spatial dimensions (from 25x25 to 50x50)
        x = self.LeakyReLU(self.conv1(x))  # Build image features
        x = self.sigmoid(self.conv2(x))    # Output layer → values between 0 and 1
        
        print("Decoder output shape:", x.shape)
        print("x_hat min: ", x.min().item(), "max: ", x.max().item())
        x_hat = x
        
        return x_hat   
