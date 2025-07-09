
clear all
close all
clc

% William Burton, 2025, University of Denver
% Rebuild anatomic model from inferred NISIM data using "marching cubes-esque" routine

recon_dir="/PATH/TO/NISIM_GRID/"
s_idx_vec=[];
model_id="2_32_2_1"; % GEOM_LATENT_REG_SIDE

% Each trial
for ss=1:1%numel(s_idx_vec)
    
    % Import NISIM data from main.py
    nisim_data_path=recon_dir+"/"+num2str(ss-1)+"_"+model_id+".txt"
    mat_in=readmatrix(nisim_data_path);
    
    % Rebuild STL from NISIM data
    [n, e, n_sm, e_sm]=rebuild_nisim_surface(mat_in);
    
    if 1==1

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        figure; hold on; axis equal;
        p(1) = patch('Faces', e, ...
                     'Vertices', n, ...
                     'FaceAlpha', 0.2,  ...
                     'FaceColor','c', ...
                     'EdgeColor','none', ...
                     'SpecularStrength',0);
        view([0,-1,0]);
        p(1) = patch('Faces', e_sm, ...
                     'Vertices', n_sm, ...
                     'FaceAlpha', 0.2,  ...
                     'FaceColor','r', ...
                     'EdgeColor','none', ...
                     'SpecularStrength',0);
        view([0,-1,0]);
        
        figure; hold on; axis equal;
        p(1) = patch('Faces', e, ...
                     'Vertices', n, ...
                     'FaceAlpha', 0.2,  ...
                     'FaceColor','c', ...
                     'EdgeColor','none', ...
                     'SpecularStrength',0);
        view([0,-1,0]);
        
        figure; hold on; axis equal;
        p(1) = patch('Faces', e_sm, ...
                     'Vertices', n_sm, ...
                     'FaceAlpha', 0.2,  ...
                     'FaceColor','r', ...
                     'EdgeColor','none', ...
                     'SpecularStrength',0);
        view([0,-1,0]);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Export geom
    path_out=recon_dir+num2str(ss-1)+"_"+model_id+"_n.txt";
    writematrix(n,path_out);
    path_out=recon_dir+num2str(ss-1)+"_"+model_id+"_e.txt";
    writematrix(e,path_out);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end








