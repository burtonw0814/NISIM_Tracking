import numpy as np
import time
import os
import uuid 
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt

from helpers import * # Computational geometry tools for processing ground truth values

# Prepare training batch for NISIM training

def get_batch(args_list):
    
    # Unpack args
    (my_data_obj, num_sampled_pts, 
        x_min_norm, x_max_norm, 
        y_min_norm, y_max_norm,
        z_min_norm, z_max_norm,
        dist_trunc, 
        hu_min, 
        hu_max, my_side, model_id, my_seed)=args_list
        
    eps=0.0
    np.random.seed(my_seed) # If using synchronous batch prep, turn off seeding (i.e., comment this out)
    tt=time.time()
    
    # Either load from stored queue or create new batch
    bx_dir="./tmp/tmp_bx_"+model_id+"/"
    by_dir="./tmp/tmp_by_"+model_id+"/"
    s_idx_dir="./tmp/tmp_s_idx_"+model_id+"/"
    max_num_temp_batches=100000
    temp_batches=os.listdir(bx_dir)
    num_temp_batches=len(temp_batches)
    
    # Load from stored batches
    if np.random.binomial(1, 0.5)==1 and num_temp_batches>0:
        idx=np.random.randint(low=0, high=num_temp_batches)
        my_f=temp_batches[idx]
        my_id=my_f.split("_")[0]
        
        bx_path=bx_dir+"/"+my_id+"_bx.npy"
        by_path=by_dir+"/"+my_id+"_by.npy"
        s_idx_path=s_idx_dir+"/"+my_id+"_s_idx.npy"
        
        bx=np.load(bx_path)
        by=np.load(by_path)
        s_idx=np.load(s_idx_path)
        
    else: # Or create new batch from scratch
        
        ##############################################################################################
        # Get data from data handler
        n, e_tet, hu_tet, e_surf, s_idx, n_ch, e_ch=my_data_obj.sample_train_inst() #
        if my_side==1: # Option to flip left/right
            n[:,0]=-1*n[:,0]
            n_ch[:,0]=-1*n_ch[:,0]
        ##############################################################################################
        
        ##############################################################################################
        # Normalize geometry into normalized space
        n_norm=n.copy()
        n_norm[:,0]=n_norm[:,0]-x_min_norm
        n_norm[:,1]=n_norm[:,1]-y_min_norm
        n_norm[:,2]=n_norm[:,2]-z_min_norm
        n_norm[:,0]=n_norm[:,0]/(np.amax((x_max_norm-x_min_norm,eps)))
        n_norm[:,1]=n_norm[:,1]/(np.amax((y_max_norm-y_min_norm,eps)))
        n_norm[:,2]=n_norm[:,2]/(np.amax((z_max_norm-z_min_norm,eps)))

        if False:
            print(np.amin(n_norm[:,0]),np.amax(n_norm[:,0]))
            print(np.amin(n_norm[:,1]),np.amax(n_norm[:,1]))
            print(np.amin(n_norm[:,2]),np.amax(n_norm[:,2]))
         
        # Normalize HU
        hu_tet[hu_tet<hu_min]=hu_min
        hu_tet[hu_tet>hu_max]=hu_max
        hu_tet=hu_tet/hu_max
        
        # Normalize convex hull nodes
        n_ch_norm=n_ch.copy()
        n_ch_norm[:,0]=n_ch_norm[:,0]-x_min_norm
        n_ch_norm[:,1]=n_ch_norm[:,1]-y_min_norm
        n_ch_norm[:,2]=n_ch_norm[:,2]-z_min_norm
        n_ch_norm[:,0]=n_ch_norm[:,0]/(np.amax((x_max_norm-x_min_norm,eps)))
        n_ch_norm[:,1]=n_ch_norm[:,1]/(np.amax((y_max_norm-y_min_norm,eps)))
        n_ch_norm[:,2]=n_ch_norm[:,2]/(np.amax((z_max_norm-z_min_norm,eps)))
        ##############################################################################################
        
        ##############################################################################################
        # Get element centroids
        e_cent=get_tet_elem_centroids(n_norm,e_tet)
        
        # Restructure surface mesh
        n_surf, e_surf=restructure_surface_mesh(n_norm, e_surf)
        
        # Normalize surface
        n_surf_norm=n_surf.copy()
        n_surf_norm[:,0]=n_surf_norm[:,0]-x_min_norm
        n_surf_norm[:,1]=n_surf_norm[:,1]-y_min_norm
        n_surf_norm[:,2]=n_surf_norm[:,2]-z_min_norm
        n_surf_norm[:,0]=n_surf_norm[:,0]/(np.amax((x_max_norm-x_min_norm,eps)))
        n_surf_norm[:,1]=n_surf_norm[:,1]/(np.amax((y_max_norm-y_min_norm,eps)))
        n_surf_norm[:,2]=n_surf_norm[:,2]/(np.amax((z_max_norm-z_min_norm,eps)))
        ##############################################################################################
        
        ##############################################################################################
        # Compute ground truth values for NISIM
        # Approach #1: Sample points, KNN search on elem centroids, check pt in tet, assign accordingly, get dist from surf
        pts_s=np.random.random(size=(int(num_sampled_pts),3)) # Sample within normlized bounding box
        
        if False:
            print(np.amin(pts_s[:,0]),np.amax(pts_s[:,0]))
            print(np.amin(pts_s[:,1]),np.amax(pts_s[:,1]))
            print(np.amin(pts_s[:,2]),np.amax(pts_s[:,2]))
        
        # KNN search between sampled points and tet element centroids
        num_neighbors=10
        knn = NearestNeighbors(n_neighbors=num_neighbors)
        knn.fit(e_cent)
        neighbors=knn.kneighbors(pts_s, return_distance=False)
        
        # Check pt in tet for 10 nearest tets, as determined by tet centroids
        in_tet_vec=np.zeros((num_sampled_pts,1))
        hu_vec=np.zeros((num_sampled_pts,1)) # HU vec determined based on HU of nearest
        for ii in range(num_sampled_pts):   
            nq=pts_s[ii,:]
            for jj in range(num_neighbors):
                
                try:
                    
                    n0=n_norm[int(e_tet[neighbors[ii,jj],0]),:]
                    n1=n_norm[int(e_tet[neighbors[ii,jj],1]),:]
                    n2=n_norm[int(e_tet[neighbors[ii,jj],2]),:]
                    n3=n_norm[int(e_tet[neighbors[ii,jj],3]),:]
                    
                    in_flag=point_inside(n0,n1,n2,n3,nq)
                    
                    if in_flag:
                        in_tet_vec[ii]=1
                        hu_vec[ii]=hu_tet[neighbors[ii,jj]] # If point is inside tet, assign HU value
                except:
                    []
                    
        # KNN search between sampled pts and surface nodes
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(n_surf)
        [d, neighbors]=knn.kneighbors(pts_s, return_distance=True)
        
        # Truncate distance data
        d[d>dist_trunc]=dist_trunc # SDF distances determined based on KNN search, better method would be point-to-triangle distances
        
        # Inner points get negative
        idx1=np.argwhere(in_tet_vec==1)
        d[idx1]=d[idx1]*-1
        
        # Check if pts in convex  hull
        idx=(Delaunay(n_ch_norm).find_simplex(pts_s)>=0).astype('uint8')
        
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(n_ch_norm[:,0], n_ch_norm[:,1], n_ch_norm[:,2], color="k")
            ax.scatter(pts_s[idx==0,0], pts_s[idx==0,1], pts_s[idx==0,2], color="b")
            ax.scatter(pts_s[idx==1,0], pts_s[idx==1,1], pts_s[idx==1,2], color="r")
            plt.show()

        # Convex hull indicators
        idx0=np.argwhere(d>0)    # Outside geom
        idx1=np.argwhere(idx==0) # Outside convex hull
        unique_idx=np.intersect1d(np.squeeze(idx0), np.squeeze(idx1)) # Outside core geom and outside convex hull
        ch_vec=0.1*np.ones((num_sampled_pts,1))
        ch_vec[unique_idx]=-1.0 # Set OUTSIDE convex hull pts to -1
        
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(n_ch_norm[:,0], n_ch_norm[:,1], n_ch_norm[:,2], color="k")
            ax.scatter(pts_s[unique_idx,0], pts_s[unique_idx,1], pts_s[unique_idx,2], color="b")
            test_idx=np.argwhere(hu_vec>0)
            ax.scatter(pts_s[test_idx,0], pts_s[test_idx,1], pts_s[test_idx,2], color="r")
            plt.show()
        ##############################################################################################
        
        ##############################################################################################
        # Reformat
        bx=pts_s.copy()
        by=np.concatenate((d,hu_vec,ch_vec),1)
        
        if False:
            print(bx.shape, by.shape)
            print(np.amin(d),np.amax(d))
            print(np.amin(hu_vec),np.amax(hu_vec))
        ##############################################################################################
        
        ##############################################################################################
        # Store as new batch or replace another if max num stored batches exceeded
        temp_batches=os.listdir(bx_dir)
        num_temp_batches=len(temp_batches)
        if num_temp_batches<max_num_temp_batches: # Either create new stored batch
            
            my_id=str(uuid.uuid1())
            bx_path=bx_dir+"/"+my_id+"_bx.npy"
            by_path=by_dir+"/"+my_id+"_by.npy"
            s_idx_path=s_idx_dir+"/"+my_id+"_s_idx.npy"
            
            np.save(bx_path, bx)
            np.save(by_path, by)
            np.save(s_idx_path, s_idx)
            
        else: # Or replace a random existing stored batch
            
            idx=np.random.randint(low=0,high=num_temp_batches)
            my_f=temp_batches[idx]
            my_id=my_f.split("_")[0]
            #print(my_id)
            
            bx_path=bx_dir+"/"+my_id+"_bx.npy"
            by_path=by_dir+"/"+my_id+"_by.npy"
            s_idx_path=s_idx_dir+"/"+my_id+"_s_idx.npy"
            
            try:
                os.remove(bx_path)
            except:
                []
                
            try:
                os.remove(by_path)
            except:
                []
                
            try:
                os.remove(s_idx_path)
            except:
                []
            
            np.save(bx_path, bx)
            np.save(by_path, by)
            np.save(s_idx_path, s_idx)
        ##############################################################################################
        
    ##############################################################################################  
    # Create positional encodings for NISIM input
    L=10
    pos_enc_dim=6*L
    bx_pos_enc=np.concatenate((  np.sin((2**0)*3.1415*bx), np.cos((2**0)*3.1415*bx),    
                                 np.sin((2**1)*3.1415*bx), np.cos((2**1)*3.1415*bx),   
                                 np.sin((2**2)*3.1415*bx), np.cos((2**2)*3.1415*bx),   
                                 np.sin((2**3)*3.1415*bx), np.cos((2**3)*3.1415*bx),   
                                 np.sin((2**4)*3.1415*bx), np.cos((2**4)*3.1415*bx),   
                                 np.sin((2**5)*3.1415*bx), np.cos((2**5)*3.1415*bx),   
                                 np.sin((2**6)*3.1415*bx), np.cos((2**6)*3.1415*bx),   
                                 np.sin((2**7)*3.1415*bx), np.cos((2**7)*3.1415*bx),   
                                 np.sin((2**8)*3.1415*bx), np.cos((2**8)*3.1415*bx),   
                                 np.sin((2**9)*3.1415*bx), np.cos((2**9)*3.1415*bx)    ), -1)
    
    #print(bx_pos_enc.shape)
    bx=np.concatenate((bx, bx_pos_enc),-1)
    #print(bx.shape)
    ##############################################################################################  
    
    ##############################################################################################  
    batch=(bx, by, s_idx)
    
    if np.random.binomial(1,0.03)==1:
        print("Batch gen time: " +str(time.time()-tt))
    ##############################################################################################  
    
    return batch
    
    
    
    
    
    
