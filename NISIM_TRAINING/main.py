import numpy as np
import time
import argparse

from data_obj import *              # Data handler
from graph_obj import *             # Wrapper class around neural net
from get_batch import *             # Get training batch method
from batch_queue_functions import * # Async batch getter object
from tmp_helpers import *           # Directory helpers

# CUDA_VISIBLE_DEVICES=0 python3 main.py --local_mode=2 --geom_idx=2  --side_idx=0 --latent_idx=3 --reg_idx=2;
# CUDA_VISIBLE_DEVICES='' python3 -m tensorboard.main --logdir=/PATH/TO/runs

if __name__ == '__main__': 

    #####################################################
    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_mode',    type=int,    required=True) # Helps distinguish paths for local/cluster training
    parser.add_argument('--geom_idx',      type=int,    required=True) # 0: femur, 1: patella, 2: tibia
    parser.add_argument('--latent_idx',    type=int,    required=True) # Specifies latent dimension of network
    parser.add_argument('--reg_idx',       type=int,    required=True) # Specifies latent regularization term
    parser.add_argument('--side_idx',      type=int,    required=True) # Option to flip from right- to left-sided anatomy
    
    args = parser.parse_args()
    local_mode=args.local_mode
    geom_idx=int(args.geom_idx)
    latent_idx=int(args.latent_idx)
    reg_idx=int(args.reg_idx)
    side_idx=int(args.side_idx)
    
    print("LOCAL MODE -- " + str(local_mode))
    print("Config args")
    print(geom_idx, latent_idx, reg_idx, side_idx)
    #####################################################
    
    #####################################################
    # Initialize net
    input_dim=3 # 3D Cartesian coords
    num_sampled_pts=17000   # Should be as high GPU memory allows
    num_train_iters=4000000 # Should be a lot 
    dist_trunc=0.2 # SDF values are truncated after exceeding a certain distance
    hu_min=0.0     # Normalize intensity values
    hu_max=1000.0  # Normalize intensity values
    
    x_min_norm_list=(-100, -50, -100)
    x_max_norm_list=( 100,  50,  100)
    y_min_norm_list=(-100, -50, -100)
    y_max_norm_list=( 100,  50,  100)
    z_min_norm_list=(-100, -50, -600)
    z_max_norm_list=( 600,  50,  100)
    
    geom_vec=(0,1,2)                    # Defines geometry to be learned
    latent_dim_vec=(32,64,128,256)      # Latent vector dimensions
    reg_vals=(0.1,0.01,0.001,0.0001)    # Magnitude of latent regularization term
    side_vals=(0,1)
    for ss in range(side_idx,side_idx+1): #len(side_vals)):
        for rr in range(reg_idx,reg_idx+1): #len(reg_vec)):
            for ll in range(latent_idx,latent_idx+1): #len(latent_dim_vec)):
                for gg in range(geom_idx,geom_idx+1): #len(geom_vec)):
                    
                    #####################################################
                    # Settings for current model
                    geom_idx=geom_vec[gg]
                    latent_dim=latent_dim_vec[ll]
                    my_reg_val=reg_vals[rr]
                    my_side=side_vals[ss]
                    model_id=str(geom_idx)+"_"+str(latent_dim)+"_"+str(rr)+"_"+str(my_side)
                    print(model_id)
                    
                    # Normalization data - defines extent of boundin boxes to sample from
                    x_min_norm=x_min_norm_list[geom_idx]
                    x_max_norm=x_max_norm_list[geom_idx]
                    y_min_norm=y_min_norm_list[geom_idx]
                    y_max_norm=y_max_norm_list[geom_idx]
                    z_min_norm=z_min_norm_list[geom_idx]
                    z_max_norm=z_max_norm_list[geom_idx]
                    #####################################################
                    
                    #####################################################
                    # Create directories for caching training batches, helps with batch prep latency
                    init_tmp_dirs(model_id)
                    #####################################################
                    
                    #####################################################
                    # Data object - handles paths to training instances + batch sampling
                    my_data_obj=Data_Obj(geom_idx, local_mode)
                    num_train_inst=my_data_obj.num_train_inst
                    print("Num train inst")
                    print(num_train_inst)
                    #####################################################
                    
                    #####################################################
                    # Initialize asynchronous batch queue -- custom wrapper class for batch prep
                    num_inst=96
                    my_seed=np.random.randint(low=0, high=10000000)
                    args_list=(my_data_obj, 
                                    num_sampled_pts,
                                    x_min_norm, x_max_norm, 
                                    y_min_norm, y_max_norm,
                                    z_min_norm, z_max_norm,
                                    dist_trunc, hu_min, hu_max, my_side, model_id, my_seed)
                    B=async_batch_queue(num_inst, args_list, get_batch)
                    #####################################################
                    
                    #####################################################
                    # Define neural net architecture
                    model_path=("./models/_model_"+model_id+"_.pth");
                    my_graph_obj=Graph_Obj(input_dim,
                            latent_dim, num_train_inst, 
                            my_reg_val, model_path, 
                            reg_val=0.0001, 
                            TB_ID=str(geom_idx)+"_"+str(latent_dim)+"_"+str(rr)+"_"+str(my_side)+"_")
                    #####################################################
                    
                    #####################################################
                    print("Beginning training loop")
                    for i in range(num_train_iters): 
                        
                        # Retrieve batches from queue that have finished processing
                        B.retrieve_batches()
                        
                        # Sample batch
                        #b=get_batch(args_list) # If using synchronous batch prep, turn off seeding in get_batch() method
                        b=B.sample_batch() # Sample batch from queue
                        
                        # Step model
                        my_graph_obj.step_model(b) # step model using sampled batch
                        time.sleep(0.001)
                        
                        # Kick off batch processor
                        B.apply_batches(args_list) 
                        
                    # Save final model and close down batch queue
                    my_graph_obj.save_model()
                    my_graph_obj=None; 
                    torch.cuda.empty_cache()
                    B.close_batch_queue()
                    ##########################################################
                    
                    
                    
                    
    






