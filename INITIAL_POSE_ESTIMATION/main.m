
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% William Burton, University of Denver, 2025
% Method for acquiring initial pose estimates in biplanar radiography based on CNN-predicted key points

% Prerequisites:
% 1. CNN-predicted (or manually annotated) 2D key points on both image planes
% 2. Template geometry triangular mesh (e.g., STL or INP format) oriented in standardized coordinate system (e.g., anatomic cosys)
% 3. IDs that match STL nodes to 2D key points
% 4. Known camera calibration data including: focal lengths, principal point, relative transform from world cosys -> camera 1,2 cosys

% Instructions:
% 1. Install YALMIP
% 2. Install MOSEK or SeDuMi as the underlying solver (MOSEK is preferred)
% 3. Set the correct solver in line 493 of opt1.m

% Other notes:
% 1. For single plane initial pose estimates, just use any built-in PnP method, either in MATLAB, Python, etc.
% 2. X-ray images, and thus key points, are assumed to be undistorted
% 3. Camera model is based on the perspective/pinhole camera mode
% 4. Camera convention includes y-axis facing down, z-axis facing outwards in the scene, x-axis facing right (when facing the scene from behind camera)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set directories
subj_data_dir="/MY/PATHS/"; % Path to X-ray images and camera calibration data (see example calibration data)
node_ids_dir="/MY/PATHS/";  % Path to node IDs which match surface mesh nodes to CNN-predicted key points (e.g., node X corresponds to key point Y)
cnn_data_dir="/MY/PATHS/";  % Path to CNN-predicted key points
geom_data_dir="/MY/PATHS/"; % Path to template surface meshes
results_dir="/MY/PATHS/";   % Where to store acquired initial pose estimates

s_idx=[]; % List of trials
geom_names={"Femur","Patella","TibFib"};
warning("off","MATLAB:MKDIR:DirectoryExists");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
`
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Import surface meshes
nodes=cell(numel(geom_names),1);
for ii=1:numel(geom_names)
    nodes_path=geom_data_dir+"s_regONODES_1_"+geom_names{ii}+".txt";
    n=readmatrix(nodes_path);
    nodes{ii}=n;
end

% Import node ids for opt1 -> Specifies which surface nodes correspond to 2D key points
node_ids=cell(numel(geom_names),1);
for ii=1:numel(geom_names)
    my_path=node_ids_dir+"node_ids_"+num2str(ii-1)+"_400.txt";
    my_data=readmatrix(my_path);
    node_ids{ii}=my_data+1; % indices were stored as zero-indexed for python but we need 1-indexed
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Iterate over trials and solve
for ii=1:numel(s_idx)
                
    % Get num frames in trial
    s_dir=cnn_data_dir+"Subject"+num2str(s_idx(ii))+"/Cam1/KP_Preds/0/";
    frames=dir(s_dir);
    frames=frames(3:end);
    frame_idx=cell(numel(frames),1);
    for jj=1:numel(frames)
        my_id=frames(jj).name;
        my_id=split(my_id,".");
        my_id=my_id{1};
        frame_idx{jj}=my_id;
    end
    
    for jj=1:numel(frame_idx) % Iterate over frames in trial
        
        for aa=1:numel(geom_names) % Each geom (femur, patela, tibia)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            cur_frame_idx=frame_idx{jj};
    
            % Import images and camera intrinsics/extrinsics
            my_path=subj_data_dir+"Subject"+num2str(s_idx(ii))+"/Cam1/Imd/"+cur_frame_idx+".jpeg";
            imd0=imread(my_path);
            imd0=cat(3, imd0, imd0, imd0);
    
            my_path=subj_data_dir+"Subject"+num2str(s_idx(ii))+"/Cam2/Imd/"+cur_frame_idx+".jpeg";
            imd1=imread(my_path);
            imd1=cat(3, imd1, imd1, imd1);
    
            my_path=subj_data_dir+"Subject"+num2str(s_idx(ii))+"/Camcal/Camcal0.txt";
            camcal0=get_camcal(my_path);
    
            my_path=subj_data_dir+"Subject"+num2str(s_idx(ii))+"/Camcal/Camcal1.txt";
            camcal1=get_camcal(my_path);
            
            % figure;
            % imshow(imd0,[]);
            % figure;
            % imshow(imd1,[]);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %num_nodes=numel(node_ids{aa});

            % Import data
            
            % Import key points
            my_path=cnn_data_dir+"Subject"+num2str(s_idx(ii))+"/Cam1/KP_Preds/"+num2str(aa-1)+"/"+cur_frame_idx+".txt";
            kp0=readmatrix(my_path);
            
            my_path=cnn_data_dir+"Subject"+num2str(s_idx(ii))+"/Cam2/KP_Preds/"+num2str(aa-1)+"/"+cur_frame_idx+".txt";
            kp1=readmatrix(my_path);
            
            % Plot key points on images
            imd0_viz=plot_kp(imd0, kp0, 2);
            imd1_viz=plot_kp(imd1, kp1, 2);
            % figure;
            % imshow(imd1_viz,[]);
            % figure;
            % imshow(imd2_viz,[]);
            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Solve opt
            cur_nodes=nodes{aa}(node_ids{aa},:);
            [Rp, Rp2, Tp]=opt1(cur_nodes, kp0, kp1, camcal0, camcal1);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Get viz
            cam_nodes0=(camcal0.R*(Rp2*nodes{aa}'+Tp)+camcal0.T)';
            cam_nodes1=(camcal1.R*(Rp2*nodes{aa}'+Tp)+camcal1.T)';
            
            proj0=project_points(cam_nodes0(1:10:end,:), ...
                                camcal0.fx, ...
                                camcal0.fy, ...
                                camcal0.cx, ...
                                camcal0.cy, ...
                                camcal0.IM(1), ...
                                camcal0.IM(2));
            
            proj1=project_points(cam_nodes1(1:10:end,:), ...
                                camcal1.fx, ...
                                camcal1.fy, ...
                                camcal1.cx, ...
                                camcal1.cy, ...
                                camcal1.IM(1), ...
                                camcal1.IM(2));
            
            imd0_viz=plot_kp(imd0_viz, proj0, 1);
            imd1_viz=plot_kp(imd1_viz, proj1, 1);
            %figure;
            %imshow(imd0_viz,[]);
            %figure;
            %imshow(imd1_viz,[]);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            % Create export dirs if they dont exist yet
            subj_dir=results_dir+"Subject"+num2str(s_idx(ii))+"/";
            dir_list={subj_dir,...
                      subj_dir+"R/",...
                      subj_dir+"T/",...
                      subj_dir+"viz0",...
                      subj_dir+"viz1/",...
                      subj_dir+"R/"+num2str(aa-1)+"/",...
                      subj_dir+"T/"+num2str(aa-1)+"/",...
                      subj_dir+"viz0/"+num2str(aa-1)+"/",...
                      subj_dir+"viz1/"+num2str(aa-1)+"/",...
                     };
            for kk=1:numel(dir_list)
                mkdir(dir_list{kk});
            end
            
            % Export transform and viz
            my_path=subj_dir+"R/"+num2str(aa-1)+"/"+cur_frame_idx+".txt";
            writematrix(Rp2,my_path);

            my_path=subj_dir+"T/"+num2str(aa-1)+"/"+cur_frame_idx+".txt";
            writematrix(Tp,my_path);

            my_path=subj_dir+"viz0/"+num2str(aa-1)+"/"+cur_frame_idx+".jpeg";
            imwrite(imd0_viz,my_path);

            my_path=subj_dir+"viz1/"+num2str(aa-1)+"/"+cur_frame_idx+".jpeg";
            imwrite(imd1_viz,my_path);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            
        end

    end

end
            

