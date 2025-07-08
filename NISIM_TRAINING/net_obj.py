import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# Actual neural net architecture of NISIM, defined via PyTorch

class Net(nn.Module):
    
    def __init__(self, input_dim, latent_dim, num_latent_vecs, devices):
        super(Net, self).__init__()
        
        self.input_dim=input_dim
        self.num_latent_vecs=num_latent_vecs # Based on number of training instances
        self.latent_dim=latent_dim
        self.devices=devices
        self.fc_dim=1024
        self.params=[];
        self.my_dict=nn.ModuleDict({})
        L=10 # Reference positional encoding definition in get_batch() method
        self.pos_enc_dim=6*L
        
        # Define layers
        self.my_dict["fc_module0"]=nn.ModuleList([
                                                torch.nn.Linear(self.input_dim+self.pos_enc_dim+self.latent_dim, self.fc_dim),
                                                torch.nn.LayerNorm(self.fc_dim),
                                                torch.nn.ReLU(inplace=True),
                                                ]).to(self.devices[0])  #
                                                
        self.my_dict["fc_module1"]=nn.ModuleList([
                                                torch.nn.Linear(self.input_dim+self.pos_enc_dim+self.fc_dim, self.fc_dim),
                                                torch.nn.LayerNorm(self.fc_dim),
                                                torch.nn.ReLU(inplace=True),
                                                ]).to(self.devices[0])  #
               
        self.my_dict["fc_module2"]=nn.ModuleList([
                                                torch.nn.Linear(self.input_dim+self.pos_enc_dim+self.fc_dim, self.fc_dim),
                                                torch.nn.LayerNorm(self.fc_dim),
                                                torch.nn.ReLU(inplace=True),
                                                ]).to(self.devices[0])  #
                   
        self.my_dict["fc_module3"]=nn.ModuleList([
                                                torch.nn.Linear(self.input_dim+self.pos_enc_dim+self.fc_dim, self.fc_dim),
                                                torch.nn.LayerNorm(self.fc_dim),
                                                torch.nn.ReLU(inplace=True),
                                                ]).to(self.devices[0])  #
                   
                   
        self.my_dict["fc_module4"]=nn.ModuleList([
                                                torch.nn.Linear(self.input_dim+self.pos_enc_dim+self.fc_dim, self.fc_dim),
                                                torch.nn.LayerNorm(self.fc_dim),
                                                torch.nn.ReLU(inplace=True),
                                                ]).to(self.devices[0])  #
                                                
        self.my_dict["fc_module5"]=nn.ModuleList([
                                                torch.nn.Linear(self.input_dim+self.pos_enc_dim+self.fc_dim, self.fc_dim),
                                                torch.nn.LayerNorm(self.fc_dim),
                                                torch.nn.ReLU(inplace=True),
                                                ]).to(self.devices[0])  #
                                     
        ############################################################################  
        # Prediction heads are split into distinct branches, found to improve training error in initial experiments         
        self.my_dict["fc_module6_a"]=nn.ModuleList([
                                                torch.nn.Linear(self.input_dim+self.pos_enc_dim+self.fc_dim, self.fc_dim),
                                                torch.nn.LayerNorm(self.fc_dim),
                                                torch.nn.ReLU(inplace=True),
                                                torch.nn.Linear(self.fc_dim, 1),
                                                torch.nn.Tanh(),
                                                ]).to(self.devices[0])  #
                                                
        self.my_dict["fc_module6_b"]=nn.ModuleList([
                                                torch.nn.Linear(self.input_dim+self.pos_enc_dim+self.fc_dim, self.fc_dim),
                                                torch.nn.LayerNorm(self.fc_dim),
                                                torch.nn.ReLU(inplace=True),
                                                torch.nn.Linear(self.fc_dim, 1),
                                                torch.nn.Tanh(),
                                                ]).to(self.devices[0])
                                                
        self.my_dict["fc_module6_c"]=nn.ModuleList([
                                                torch.nn.Linear(self.input_dim+self.pos_enc_dim+self.fc_dim, self.fc_dim),
                                                torch.nn.LayerNorm(self.fc_dim),
                                                torch.nn.ReLU(inplace=True),
                                                torch.nn.Linear(self.fc_dim, 1),
                                                torch.nn.Tanh(),
                                                ]).to(self.devices[0])
        ############################################################################   
            
        ############################################################################                
        # Initialize latent vectors that correspond to particular training instances
        self.latent_vecs=[];
        for ii in range(self.num_latent_vecs):
            new_latent_vec=nn.Parameter(torch.randn(1,self.latent_dim)*0.01, requires_grad=True) 
            self.latent_vecs.append(new_latent_vec)
        ############################################################################ 
        
        ############################################################################ 
        self.myparameters = nn.ParameterList(self.latent_vecs)
        self.myparameters.to(self.devices[0]) 
        print('MY PARAM LIST: ' + str(len(self.myparameters)))
        ############################################################################ 
        
        return
        
    def forward_train(self, x_in, latent_idx):
        
        # Pull appropriate latent vector
        latent_tens=self.latent_vecs[latent_idx].repeat(x_in.size(0),1) # Assume single training instance per batch
        
        x=torch.cat((x_in,latent_tens),-1)
        
        #print(x_in.size(), latent_tens.size(), x.size())
        
        for ii in range(len(self.my_dict["fc_module0"])):
            x=self.my_dict["fc_module0"][ii](x)
        x=torch.cat((x, x_in), -1)
        #x=torch.cat((x, x_in, latent_tens), -1) # Could also try concatenating latent vector here
        
        for ii in range(len(self.my_dict["fc_module1"])):
            x=self.my_dict["fc_module1"][ii](x)
        x=torch.cat((x, x_in), -1)
        
        for ii in range(len(self.my_dict["fc_module2"])):
            x=self.my_dict["fc_module2"][ii](x)
        x=torch.cat((x, x_in), -1)

        for ii in range(len(self.my_dict["fc_module3"])):
            x=self.my_dict["fc_module3"][ii](x)
        x=torch.cat((x, x_in), -1)
        
        for ii in range(len(self.my_dict["fc_module4"])):
            x=self.my_dict["fc_module4"][ii](x)
        x=torch.cat((x, x_in), -1)
        
        for ii in range(len(self.my_dict["fc_module5"])):
            x=self.my_dict["fc_module5"][ii](x)
        x=torch.cat((x, x_in), -1)
        
        #########################################################
        x0=x
        for ii in range(len(self.my_dict["fc_module6_a"])):
            x0=self.my_dict["fc_module6_a"][ii](x0)
            
        x1=x
        for ii in range(len(self.my_dict["fc_module6_b"])):
            x1=self.my_dict["fc_module6_b"][ii](x1)
            
        x2=x
        for ii in range(len(self.my_dict["fc_module6_c"])):
            x2=self.my_dict["fc_module6_c"][ii](x2)
            
        x=torch.cat((x0,x1,x2),-1)
        #########################################################
        
        return x, latent_tens

    def forward(self, x_in, latent_tens):
        
        x=torch.cat((x_in,latent_tens),-1)
        
        for ii in range(len(self.my_dict["fc_module0"])):
            x=self.my_dict["fc_module0"][ii](x)
        x=torch.cat((x, x_in), -1)
        
        for ii in range(len(self.my_dict["fc_module1"])):
            x=self.my_dict["fc_module1"][ii](x)
        x=torch.cat((x, x_in), -1)
        
        for ii in range(len(self.my_dict["fc_module2"])):
            x=self.my_dict["fc_module2"][ii](x)
        x=torch.cat((x, x_in), -1)

        for ii in range(len(self.my_dict["fc_module3"])):
            x=self.my_dict["fc_module3"][ii](x)
        x=torch.cat((x, x_in), -1)
        
        for ii in range(len(self.my_dict["fc_module4"])):
            x=self.my_dict["fc_module4"][ii](x)
        x=torch.cat((x, x_in), -1)

        for ii in range(len(self.my_dict["fc_module5"])):
            x=self.my_dict["fc_module5"][ii](x)
        x=torch.cat((x, x_in), -1)
        
        #########################################################
        x0=x
        for ii in range(len(self.my_dict["fc_module6_a"])):
            x0=self.my_dict["fc_module6_a"][ii](x0)
            
        x1=x
        for ii in range(len(self.my_dict["fc_module6_b"])):
            x1=self.my_dict["fc_module6_b"][ii](x1)
            
        x2=x
        for ii in range(len(self.my_dict["fc_module6_c"])):
            x2=self.my_dict["fc_module6_c"][ii](x2)
            
        x=torch.cat((x0,x1,x2),-1)
        #########################################################
        
        return x
            
        
        
        
