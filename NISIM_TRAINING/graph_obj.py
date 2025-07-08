import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from net_obj import *

# NN utilities wrapper around actual neural net architecture

class Graph_Obj():
    
    def __init__(self, input_dim, latent_dim, 
                       num_train_inst, latent_reg_val, 
                       model_path, 
                       load=False, reg_val=0.0001, 
                       TB_ID=0, L_rate=0.000005):
        
        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.num_train_inst=num_train_inst # Determines number of trained latent vectors
        self.latent_reg_val=latent_reg_val
        self.model_path=model_path
        self.display_step=1000
        self.save_step=1000
        self.ct=0
        self.mixed_prec=True
        self.TB_ID=TB_ID
        self.L_rate=L_rate
        self.reg_val=reg_val # Use generic regularization for actual network parameters
        self.devices = (torch.device("cuda:0"), torch.device("cuda:1"))
        print("DEVICES")
        print(self.devices)
        
        if load==False:
            self.my_net=Net(self.input_dim, 
                            self.latent_dim,
                            self.num_train_inst,
                            self.devices)
        else:
            self.load_model()

        self.my_loss=self.loss
        
        print('DEVICE INFO')
        print('CURRENT DEVICES: ' + str(self.devices))
        print('CUDA AVAILABLE: ' + str(torch.cuda.is_available()))
        device_ids=range(torch.cuda.device_count())
        print(device_ids)
        
        self.opt = torch.optim.Adam(self.my_net.parameters(), 
                                                lr=self.L_rate, 
                                                weight_decay=self.reg_val)
        self.latent_opt = torch.optim.Adam(self.my_net.latent_vecs, 
                                                lr=self.L_rate, 
                                                weight_decay=self.reg_val)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 
                                                                mode='min',
                                                                factor=0.2, 
                                                                patience=100000,
                                                                verbose=True,
                                                                threshold=1e-5, 
                                                                min_lr=3e-7)                                                       
        self.sched_latent = torch.optim.lr_scheduler.ReduceLROnPlateau(self.latent_opt, 
                                                                mode='min',
                                                                factor=0.2, 
                                                                patience=100,
                                                                verbose=True,
                                                                threshold=1e-5, 
                                                                min_lr=3e-7)  
        
        self.writer=SummaryWriter('runs/TB/' + str(self.TB_ID))
        
        if self.mixed_prec:
            self.my_scaler=torch.cuda.amp.GradScaler()
        
        return
        
    def loss(self, pred, by, latent_vec, return_comps=False):)
        l1=self.latent_reg_val*torch.mean(torch.pow(torch.cat((self.my_net.latent_vecs), 0),2)) # Access original parameter
        l_sdf=10.0*F.l1_loss(pred[:,0],by[:,0])
        l_hu=F.l1_loss(pred[:,1],by[:,1])
        l_ch=F.l1_loss(pred[:,2],by[:,2])
        if return_comps:
            return l_sdf+l_hu+l_ch, l1, l_sdf, l_hu, l_ch
        else:
            return l_sdf+l_hu+l_ch+l1
        
    def loss_latent_opt(self, pred, by, latent_vec, return_comps=False):
        l1=0.0*torch.mean(torch.pow(torch.cat((self.my_net.latent_vecs), 0),2)) # Access original parameter
        l_sdf=10.0*F.l1_loss(pred[:,0],by[:,0])#  10.0*torch.mean(torch.pow(pred[:,0]-by[:,0],2))  #   
        l_hu=F.l1_loss(pred[:,1],by[:,1])
        l_ch=F.l1_loss(pred[:,2],by[:,2])
        if return_comps:
            return l_sdf+l_hu+l_ch, l1, l_sdf, l_hu, l_ch
        else:
            return l_sdf+l_hu+l_ch+l1
            
    def step_model(self, batch):
    
        # For learning latent vectors and network params as part of training process
        
        # Unpack batch data
        t1=time.time()
        bx=torch.from_numpy(batch[0]).float().to(self.devices[0])
        by=torch.from_numpy(batch[1]).float().to(self.devices[0])
        s_idx=batch[2]
        
        if self.mixed_prec: # Mixed precision training
            self.opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred, latent_vec=self.my_net.forward_train(bx, s_idx)
                loss_compute=self.my_loss(pred, by, latent_vec)    
            self.my_scaler.scale(loss_compute).backward()
            self.my_scaler.step(self.opt)
            self.my_scaler.update()
            self.sched.step(loss_compute)#
        else:
            self.opt.zero_grad()
            pred, latent_vec=self.my_net.forward_train(bx, s_idx)
            loss_compute=self.my_loss(pred, by, latent_vec) 
            loss_compute.backward()
            torch.nn.utils.clip_grad_norm_(self.my_net.parameters(), 5.0)
            self.opt.step()
            self.sched.step(loss_compute)#
        
        if self.ct%self.display_step==0 or self.ct<10:
            self.add_summary(pred, by, latent_vec)
            print(str(time.time()-t1) + ' Seconds'); print('');
        if self.ct%self.save_step==0:
            self.save_model()
        if np.random.binomial(1, 0.02)==1:
            print(str(time.time()-t1) + ' Seconds'); print('');
        self.ct+=1
        
        return self.ct
        
    def step_latent(self, batch):
    
        # For learning latent vec given fixed network params and a new instance
        
        # Unpack batch data
        t1=time.time()
        bx=torch.from_numpy(batch[0]).float().to(self.devices[0])
        by=torch.from_numpy(batch[1]).float().to(self.devices[0])
        s_idx=0
        
        if self.mixed_prec: # Mixed precision training
            self.latent_opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred, latent_vec=self.my_net.forward_train(bx, s_idx)
                loss_compute=self.loss_latent_opt(pred, by, latent_vec)    
            self.my_scaler.scale(loss_compute).backward()
            self.my_scaler.step(self.latent_opt)
            self.my_scaler.update()
            self.sched_latent.step(loss_compute)#
        else:
            self.latent_opt.zero_grad()
            pred, latent_vec=self.my_net.forward_train(bx, s_idx)
            loss_compute=self.loss_latent_opt(pred, by, latent_vec)    
            loss_compute.backward()
            torch.nn.utils.clip_grad_norm_(self.my_net.parameters(), 5.0)
            self.latent_opt.step()
            self.sched_latent.step(loss_compute)#
        
        latent_detach=np.squeeze(latent_vec.cpu().detach().numpy()[0,:])
        
        if self.ct%10==0 or self.ct<10:
            print(self.ct, loss_compute, self.latent_opt.param_groups[0]['lr'])
        self.ct+=1
        
        return latent_detach
        
    def save_model(self, path=None):
        if path==None:  
            path=self.model_path
        checkpoint = {'model': self.my_net,
                      'state_dict': self.my_net.state_dict(),
                      'optimizer' : self.opt.state_dict()}#,
                      #'scheduler' : self.sched.state_dict()}
        torch.save(checkpoint, path)     
        print("Saved model to " + path)    
        return
        
    def load_model(self, path=None):
        if path==None:  
            path=self.model_path
        
        print('Loading model ' + str(path)) 
        checkpoint = torch.load(path)
        self.my_net = checkpoint['model']
        self.my_net.load_state_dict(checkpoint['state_dict']) 
        
        self.my_net.eval()
        
        self.my_net.devices=self.devices
        print("NET DEVICES")
        print(self.my_net.devices)
        
        return  
        
    def get_prediction(self, bx, s_idx):
        bx=torch.from_numpy(bx).float().to(self.devices[0])
           
        if self.mixed_prec:
            # Mixed precision inference   
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred, latent_tens=self.my_net.forward_train(bx, s_idx)
        else:
            pred, latent_tens=self.my_net.forward_train(bx, s_idx)
        
        return pred.cpu().detach().numpy(), latent_tens.cpu().detach().numpy()
        
    def add_summary(self, pred, by, latent_vec):
        
        # Loss components
        l0, l1, l_sdf, l_hu, l_ch=self.loss(pred, by, latent_vec, return_comps=True)
        self.writer.add_scalar('l0', l0, global_step=self.ct)
        self.writer.add_scalar('l1', l1, global_step=self.ct) 
        self.writer.add_scalar('l', l0+l1, global_step=self.ct)
        self.writer.add_scalar('norm_latent_', torch.mean(torch.pow(torch.cat((self.my_net.latent_vecs), 0),2)), global_step=self.ct)
        latent_tens_grads=torch.cat((    [self.my_net.latent_vecs[i].grad for i in range(len(self.my_net.latent_vecs)) ]   ), 0)
        self.writer.add_scalar('norm_latent_grad_', torch.mean(torch.pow(latent_tens_grads,2)), global_step=self.ct)
        self.writer.add_scalar('l_sdf', l_sdf, global_step=self.ct)
        self.writer.add_scalar('l_hu',  l_hu,  global_step=self.ct) 
        self.writer.add_scalar('l_ch',  l_ch,  global_step=self.ct) 
        self.writer.add_scalar('learning_rate', self.opt.param_groups[0]['lr'], global_step=self.ct)
        self.writer.add_scalar('latent_reg_val',   self.latent_reg_val,  global_step=self.ct) 
        self.latent_reg_val
        
        return     

        
