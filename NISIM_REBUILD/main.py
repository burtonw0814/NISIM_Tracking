import os
import numpy as np
import torch

from config import *
from net_obj import *

# William Burton, 2025, University of Denver
# Import specficied latent vector (e.g., from registration) and rebuild grid of NISIM data for STL reconstruction
# This script is followed by nisim_reconstruction.m

if __name__ == '__main__': 

    #########################################################################################
    # Paths
    model_dir="/path/to/models/"
    opt2_dir="/path/to/latent/"
    latent_dir_out="/path/to/export/"
    #########################################################################################
    
    #########################################################################################
    # Model configuration
    reg_idx=2
    my_side=0
    geom_idx=0
    latent_dim=64
    x_min_norm_list=(-100, -50, -100) # Must match that used in NISIM training
    x_max_norm_list=( 100,  50,  100)
    y_min_norm_list=(-100, -50, -100)
    y_max_norm_list=( 100,  50,  100)
    z_min_norm_list=(-100, -50, -600)
    z_max_norm_list=( 600,  50,  100)
    dist_trunc=0.2
    hu_min=0.0
    hu_max=1000.0
    
    # Norm data
    x_min_norm=x_min_norm_list[geom_idx]
    x_max_norm=x_max_norm_list[geom_idx]
    y_min_norm=y_min_norm_list[geom_idx]
    y_max_norm=y_max_norm_list[geom_idx]
    z_min_norm=z_min_norm_list[geom_idx]
    z_max_norm=z_max_norm_list[geom_idx]
    
    s_list=() # List of trials
    #########################################################################################
    
    #########################################################################################
    # Load model
    model_suffix=str(geom_idx)+"_"+str(latent_dim)+"_"+str(reg_idx)+"_"+str(my_side)
    model_path=model_dir+"/_model_"+model_suffix+"_.pth";
    print('Loading model ' + str(model_path)) 
    checkpoint = torch.load(model_path, map_location="cpu") #torch.device("cpu")
    my_net = checkpoint['model']
    my_net.load_state_dict(checkpoint['state_dict'])    
    my_net.eval()
    my_net=my_net.to(torch.device("cuda:0")) 
    #########################################################################################
    
    for ss in range(len(s_list)): # Each trial
        
        #########################################################################################
        s_idx=s_list[ss]
        
        # Get frames
        latent_dir=opt2_dir+str(cfg_idx)+"/Subject"+str(s_idx)+"/LATENT/"+str(geom_idx)+"/";
        frames=os.listdir(latent_dir)
        frames.sort()
        frame_ids=[]     
        for ii in range(len(frames)):
            cur_f=frames[ii].split(".")[0]
            frame_ids.append(cur_f)
        frame_ids.sort()
        #########################################################################################
        
        # Each frame in trial
        for ff in range(len(frame_ids)):
            
            #########################################################################################
            # Import latent code (or define a random one)
            fid=frame_ids[ff]
            latent_path=latent_dir+fid+".txt"
            print(latent_path)
            latent_vec=np.loadtxt(latent_path)
            #########################################################################################
            
            ##################################################################
            # Create regular grid as sample pts
            num_x_steps=128#184
            num_y_steps=128#256#184
            num_z_steps=128#256
            x_inc=1.0/(num_x_steps-1)
            y_inc=1.0/(num_y_steps-1)
            z_inc=1.0/(num_z_steps-1)
            
            incs_x=np.linspace(0,1,num_x_steps)
            incs_y=np.linspace(0,1,num_x_steps) 
            incs_z=np.linspace(0,1,num_x_steps)
            x_idx_vec=np.expand_dims(np.array(list(range(0,num_x_steps))),-1)
            y_idx_vec=np.expand_dims(np.array(list(range(0,num_y_steps))),-1)
            z_idx_vec=np.expand_dims(np.array(list(range(0,num_z_steps))),-1)
            
            num_grid_pts=num_x_steps*num_y_steps*num_z_steps
            grid=np.zeros((num_grid_pts,3))
            ct=0
            pts_list=[]
            idx_list=[]
            for kk in range(num_z_steps):
                cur_z_inc=z_inc*kk
                all_z_pts=cur_z_inc*np.ones(num_x_steps)

                cur_z_idx=kk
                all_z_idx=cur_z_idx*np.ones(num_x_steps)
                for jj in range(num_y_steps):
                    #for ii in range(1): #num_x_steps):
                    cur_y_inc=y_inc*jj
                    all_y_pts=cur_y_inc*np.ones(num_x_steps)
                    new_pts=np.stack((incs_x,all_y_pts,all_z_pts),-1)
                    pts_list.append(new_pts)

                    cur_y_idx=jj
                    all_y_idx=cur_y_idx*np.ones(num_x_steps)
                    new_idx=np.stack((np.squeeze(x_idx_vec),all_y_idx,all_z_idx),-1)     
                    idx_list.append(new_idx)
                    
            all_pts=np.concatenate(pts_list,0)      
            all_idx=np.concatenate(idx_list,0)
            print(all_pts.shape)
            ##################################################################
            
            ##################################################################
            # Inference on entire grid to create synthetic CT
            inf_batch_size=20000
            num_inf_batches=int(float(num_grid_pts)/float(inf_batch_size)+1.0)
            all_grid_data=[];
            ct=0
            for ii in range(num_inf_batches):
                
                #########################################################################################
                # Perform inference on subset of ts due to memory constraints
                if ct+inf_batch_size<num_grid_pts:
                    cur_pts=all_pts[ct:ct+inf_batch_size,:]
                    ct=ct+inf_batch_size
                else:
                    cur_pts=all_pts[ct:,:]
                    ct=ct+inf_batch_size
                #########################################################################################
                
                #########################################################################################
                L=10
                pos_enc_dim=6*L
                bx_pos_enc=np.concatenate((  np.sin((2**0)*3.1415*cur_pts), np.cos((2**0)*3.1415*cur_pts),    
                                             np.sin((2**1)*3.1415*cur_pts), np.cos((2**1)*3.1415*cur_pts),   
                                             np.sin((2**2)*3.1415*cur_pts), np.cos((2**2)*3.1415*cur_pts),   
                                             np.sin((2**3)*3.1415*cur_pts), np.cos((2**3)*3.1415*cur_pts),   
                                             np.sin((2**4)*3.1415*cur_pts), np.cos((2**4)*3.1415*cur_pts),   
                                             np.sin((2**5)*3.1415*cur_pts), np.cos((2**5)*3.1415*cur_pts),   
                                             np.sin((2**6)*3.1415*cur_pts), np.cos((2**6)*3.1415*cur_pts),   
                                             np.sin((2**7)*3.1415*cur_pts), np.cos((2**7)*3.1415*cur_pts),   
                                             np.sin((2**8)*3.1415*cur_pts), np.cos((2**8)*3.1415*cur_pts),   
                                             np.sin((2**9)*3.1415*cur_pts), np.cos((2**9)*3.1415*cur_pts)    ), -1)
                cur_pts_in=np.concatenate((cur_pts, bx_pos_enc),-1)
                #########################################################################################
                
                #########################################################################################
                # Inference
                bx=torch.from_numpy(cur_pts_in).float().to(torch.device("cuda:0"))
                latent_tens=torch.from_numpy(latent_vec).float().to(torch.device("cuda:0")).repeat(bx.size(0),1)
                
                with torch.autocast(device_type="cuda", dtype=torch.float16): # Assume NISIM was trained using mixed prec
                    pred=my_net.forward(bx, latent_tens)
                pred=pred.cpu().detach().numpy()
                
                # Unnormalize pts
                cur_pts_unnorm=cur_pts.copy()
                cur_pts_unnorm[:,0]=cur_pts_unnorm[:,0]*(x_max_norm-x_min_norm)
                cur_pts_unnorm[:,1]=cur_pts_unnorm[:,1]*(y_max_norm-y_min_norm)
                cur_pts_unnorm[:,2]=cur_pts_unnorm[:,2]*(z_max_norm-z_min_norm)
                cur_pts_unnorm[:,0]=cur_pts_unnorm[:,0]+x_min_norm
                cur_pts_unnorm[:,1]=cur_pts_unnorm[:,1]+y_min_norm
                cur_pts_unnorm[:,2]=cur_pts_unnorm[:,2]+z_min_norm
                
                # Unnormalize preds
                sd_pred=pred[:,:1]
                hu_pred=pred[:,1:2]
                ch_pred=pred[:,2:3]
                hu_pred=hu_pred*hu_max
                
                # Store unnormalized pts and preds
                cur_grid_data=np.concatenate((cur_pts_unnorm,sd_pred,hu_pred,ch_pred),-1)
                all_grid_data.append(cur_grid_data)
                #########################################################################################
                
            #########################################################################################
            # Reformat all data from batches
            all_grid_data=np.concatenate(all_grid_data,0)
            all_grid_data=np.concatenate((all_idx,all_grid_data),1)
            ##################################################################
            
            ##################################################################
            # Export grid
            mkdir_list=(latent_dir_out+str(cfg_idx),
                        latent_dir_out+str(cfg_idx)+"/Subject"+str(s_idx)+"/"
                        )
            for ii in range(len(mkdir_list)):
                try:
                    os.mkdir(mkdir_list[ii])
                except:
                    []
            latent_path_out=latent_dir_out+str(cfg_idx)+"/Subject"+str(s_idx)+"/"+str(geom_idx)+"_"+fid+".txt";
            print(latent_path_out)
            np.savetxt(latent_path_out, all_grid_data)
            ###############################################################
                    
