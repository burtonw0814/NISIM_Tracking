import os
import numpy as np
import time
import matplotlib.pyplot as plt
import imageio

# Object for importing raw data used for training NISIMs

class Data_Obj():
    
    def __init__(self, geom_idx, local_mode, train_mode=True):
        
        # Currently configured for internal dataset
        
        # Options for data for extended usage include:
        # TotalSegmentor training dataset: https://pubs.rsna.org/doi/full/10.1148/ryai.230024
        # New Mexico dataset: https://nmdid.unm.edu/
        # Spine1K: https://github.com/MIRACLE-Center/CTSpine1K
        # Pelvis1k: https://github.com/MIRACLE-Center/CTPelvic1K
        # other open source medical image data
        # Also using closer to 100 or more training instances is recommended
        
        self.el="2"
        self.train_s_idx=list()
        self.test_s_idx=()
        self.geom_names=("Femur","Patella","Tibia")
        self.geom_idx=geom_idx
        self.local_mode=int(local_mode)
        self.cur_geom_name=self.geom_names[self.geom_idx]
        self.num_train_inst=len(self.train_s_idx)     
        self.num_test_inst=len(self.test_s_idx)     
        
        # For handling local vs. cluster training
        if self.local_mode==0:
            self.data_dir="/PATH/TO/subjects/"   
            self.convex_hull_dir="/PATH/TO/convex_hulls/"   
        elif local_mode==1:
            self.data_dir="/PATH/TO/subjects/"   
            self.convex_hull_dir="/PATH/TO/convex_hulls/"  
        else:
            self.data_dir="/PATH/TO/subjects/"   
            self.convex_hull_dir="/PATH/TO/convex_hulls/"   
        
        print("DATA DIR")
        print(self.data_dir)  
        
        return
        
    def sample_train_inst(self):
    
        # Training data is in format of material-mapped tet meshes -- a typical format used for development of SSIMS
        
        s_idx=np.random.randint(low=0, high=len(self.train_s_idx))
        s_id=str(self.train_s_idx[s_idx])
        while len(s_id)<2:
            s_id="0"+s_id        
        
        data_root=self.data_dir+"/S"+s_id+"/"
        n_path=        data_root+"s"+s_id+"_"+self.cur_geom_name+"_"+str(self.el)+"_ns_tet_reg_fine.txt"
        tet_path=      data_root+"s"+s_id+"_"+self.cur_geom_name+"_"+str(self.el)+"_es_tet_reg_fine.txt"
        hu_path=       data_root+"s"+s_id+"_"+self.cur_geom_name+"_"+str(self.el)+"_tet_reg_hu.txt"
        surf_tris_path=data_root+"s"+s_id+"_"+self.cur_geom_name+"_"+str(self.el)+"_reg_fine_tet_surf_tris.txt"
        
        n=np.loadtxt(         n_path,         delimiter=",")
        e_tet=np.loadtxt(     tet_path,       delimiter=",")
        e_tet=e_tet-1 # Subtract 1 if needed to get zero-indexed
        hu_tet=np.loadtxt(    hu_path,        delimiter=",")
        e_surf=np.loadtxt( surf_tris_path, delimiter=",")
        e_surf=e_surf-1 # Subtract 1 if needed to get zero-indexed
        
        # Import convex hull
        n_ch_path=self.convex_hull_dir+"/s"+s_id+"_"+self.cur_geom_name+"_n.txt"
        e_ch_path=self.convex_hull_dir+"/s"+s_id+"_"+self.cur_geom_name+"_e.txt"
        
        n_ch=np.loadtxt(n_ch_path, delimiter=",")
        e_ch=np.loadtxt(e_ch_path, delimiter=",")
        e_ch=e_ch-1  # Subtract 1 if needed to get zero-indexed
            
        if False:
            print(n.shape, e_tet.shape, hu_tet.shape, e_surf.shape)
            print(np.amin(hu_tet), np.amax(hu_tet))
            
        # s_idx:   scalar
        # n:       [N,3] - tet mesh nodes
        # e_tet:   [E,4] - tet meshes elements
        # hu_tet:  [E,1] - tet intensity values
        # e_surf:  [M,3] - tet surface tri elements
        # n_ch:    [C,3] - convex hull nodes
        # e_ch:    [B,3] - convex hull elements
        
        return n, e_tet, hu_tet, e_surf, s_idx, n_ch, e_ch
        
    def sample_test_inst(self, test_idx_in):
        
        s_id=str(self.test_s_idx[test_idx_in])
        while len(s_id)<2:
            s_id="0"+s_id        
        
        data_root=self.data_dir+"/S"+s_id+"/"
        n_path=        data_root+"s"+s_id+"_"+self.cur_geom_name+"_"+str(self.el)+"_ns_tet_reg_fine.txt"
        tet_path=      data_root+"s"+s_id+"_"+self.cur_geom_name+"_"+str(self.el)+"_es_tet_reg_fine.txt"
        hu_path=       data_root+"s"+s_id+"_"+self.cur_geom_name+"_"+str(self.el)+"_tet_reg_hu.txt"
        surf_tris_path=data_root+"s"+s_id+"_"+self.cur_geom_name+"_"+str(self.el)+"_reg_fine_tet_surf_tris.txt"
        
        n=np.loadtxt(         n_path,         delimiter=",")
        e_tet=np.loadtxt(     tet_path,       delimiter=",")
        e_tet=e_tet-1  # Subtract 1 if needed to get zero-indexed
        hu_tet=np.loadtxt(    hu_path,        delimiter=",")
        e_surf=np.loadtxt( surf_tris_path, delimiter=",")
        e_surf=e_surf-1  # Subtract 1 if needed to get zero-indexed
        
        # Import convex hull
        n_ch_path=self.convex_hull_dir+"/s"+s_id+"_"+self.cur_geom_name+"_n.txt"
        e_ch_path=self.convex_hull_dir+"/s"+s_id+"_"+self.cur_geom_name+"_e.txt"
        
        n_ch=np.loadtxt(n_ch_path, delimiter=",")
        e_ch=np.loadtxt(e_ch_path, delimiter=",")
        e_ch=e_ch-1  # Subtract 1 if needed to get zero-indexed
            
        if False:
            print(n.shape, e_tet.shape, hu_tet.shape, e_surf.shape)
            print(np.amin(hu_tet), np.amax(hu_tet))
        
        return n, e_tet, hu_tet, e_surf, n_ch, e_ch

        
