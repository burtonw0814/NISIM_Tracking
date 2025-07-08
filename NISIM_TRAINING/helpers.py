import numpy as np
import time

# Computational geometry primitives for creating training batches

def get_tet_elem_centroids(n, e_tet):

    # Get centroids of tet elements in tet mesh

    num_nodes=n.shape[0]
    num_elems=e_tet.shape[0]
    
    e_cent=np.zeros((num_elems,3))
    for ii in range(num_elems):
        n1_idx=int(e_tet[ii,0])
        n2_idx=int(e_tet[ii,1])
        n3_idx=int(e_tet[ii,2])
        n4_idx=int(e_tet[ii,3])
        
        n1=n[n1_idx,:]
        n2=n[n2_idx,:]
        n3=n[n3_idx,:]
        n4=n[n4_idx,:]
        
        e_cent[ii,:]=(n1+n2+n3+n4)/4.0
    
    return e_cent
    
def restructure_surface_mesh(n, surf_tris):

    # Takes nodes and list of tri elements that define a tet mesh surf, creates renumbered and filtered surf mesh

    filt_nodes_idx=np.unique(surf_tris.flatten())
    
    num_new_nodes=filt_nodes_idx.shape[0]
    num_surf_tris=surf_tris.shape[0]
    
    new_nodes=np.zeros((num_new_nodes,3))
    new_elems=np.zeros((num_surf_tris,3))
    e_alloc=np.zeros((num_surf_tris,3))
    for ii in range(num_new_nodes):
        
        old_node_idx=int(filt_nodes_idx[ii])
        new_nodes[ii,:]=n[old_node_idx,:]
        
        idx=np.argwhere(surf_tris==old_node_idx)
        for jj in range(idx.shape[0]):
            my_row=idx[jj,0]
            my_col=idx[jj,1]
            alloc_flag=e_alloc[my_row, my_col]
            if alloc_flag==0: # Only update new element if that spot hasnt been allocated
                e_alloc[my_row, my_col]=1
                new_elems[my_row, my_col]=ii
        
    return new_nodes, new_elems
    
# https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not
# Dorians answer
def tet_coord(A,B,C,D):
    v1 = B-A ; v2 = C-A ; v3 = D-A
    # mat defines an affine transform from the tetrahedron to the orthogonal system
    mat = np.array((v1,v2,v3)).T
    # The inverse matrix does the opposite (from orthogonal to tetrahedron)
    M1 = np.linalg.inv(mat)
    return(M1) 
    
def point_inside(v1,v2,v3,v4,p):
    # Find the transform matrix from orthogonal to tetrahedron system
    M1=tet_coord(v1,v2,v3,v4)
    # apply the transform to P
    newp = M1.dot(p-v1)
    # perform test
    return (np.all(newp>=0) and np.all(newp <=1) and np.sum(newp)<=1)
        
    
       
       
