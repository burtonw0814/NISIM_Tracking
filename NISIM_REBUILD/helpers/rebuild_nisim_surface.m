function [n_surf,e_surf,n_surf_sm,e_surf_sm]=rebuild_nisim_surface(mat_in)

    % William Burton, 2025, University of Denver
    % Rebuild surface mesh from NISIM data
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mat_in(:,1:3)=mat_in(:,1:3)+1;   % Python is 0-indexed, we want 1-indexed
    idx_mat=mat_in(:,1:3);           % Grid indices
    pts=mat_in(:,4:end);             % Points and NN outputs
    sdf_col=4;                       % Column which contains SDF value
    hu_col=5;                        % Column that contains HU value
    ch_col=6;                        % Column that contains convex hull indicator
    x_dim=128;                       % Must match params in main.py
    y_dim=128;                       % Must match params in main.py
    z_dim=128;                       % Must match params in main.py
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Get x/y/z-min/max indices for tightest bounding box around geom borders
    idx=find(pts(:,sdf_col)<0);
    x_vals=idx_mat(idx,1);
    y_vals=idx_mat(idx,2);
    z_vals=idx_mat(idx,3);
    
    min_x_idx=min(x_vals)-1;
    max_x_idx=max(x_vals)+1;
    min_y_idx=min(y_vals)-1;
    max_y_idx=max(y_vals)+1;
    min_z_idx=min(z_vals)-1;
    max_z_idx=max(z_vals)+1;

    idx_ch=find(pts(:, ch_col)>0);
    ch_pts=pts(idx_ch,1:3);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if 1==0
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Scatter plot of SDF fields
        figure;
        sdf_pts=pts(:,sdf_col);
        cmap=colormap;
        num_cols=size(cmap,1);
        sdf_col_idx=(sdf_pts-min(sdf_pts))/(max(sdf_pts)-min(sdf_pts))*num_cols;
        sdf_col_idx(sdf_col_idx<1)=1;
        sdf_col_idx(sdf_col_idx>num_cols)=num_cols;
        sdf_col_idx=round(sdf_col_idx);
        sdf_cols=cmap(sdf_col_idx,:);
        y_min=min(idx_mat(:,2));
        y_max=max(idx_mat(:,2));
        y_range=y_max-y_min;
        plot_idx=find(idx_mat(:,2)<(y_min+0.55*(y_max-y_min)));
        plot_idx=plot_idx([1:10:numel(plot_idx)]);
        %[1:100:size(idx_mat,1)];
        scatter3(idx_mat(plot_idx,1), ...
                    idx_mat(plot_idx,2), ...
                    idx_mat(plot_idx,3), ...
                    30, ...
                    cmap(sdf_col_idx(plot_idx),:), ..., ...
                    'filled',...
                    "MarkerFaceAlpha",0.99, ...
                    "MarkerEdgeAlpha",0.99);
        view([0,1,0]);
        axis off; axis equal;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Scatter plot of intensity fields
        figure;
        hu_pts=pts(:,hu_col);
        col_min=[0,0,0];
        col_max=[0.7,0.7,0.8];
        num_cols=255;
        col_incs=[0:1:(num_cols-1)]'/(num_cols-1);
        cmap=col_min+col_incs.*(col_max-col_min);
        num_cols=size(cmap,1);
        hu_col_idx=(hu_pts-min(hu_pts))/(max(hu_pts)-min(hu_pts))*num_cols;
        hu_col_idx(hu_col_idx<1)=1;
        hu_col_idx(hu_col_idx>num_cols)=num_cols;
        hu_col_idx=round(hu_col_idx);
        hu_cols=cmap(hu_col_idx,:);
        y_min=min(idx_mat(:,2));
        y_max=max(idx_mat(:,2));
        y_range=y_max-y_min;
        plot_idx=find(idx_mat(:,2)<(y_min+0.53*(y_max-y_min)));
        plot_idx=plot_idx([1:5:numel(plot_idx)]);
        scatter3(idx_mat(plot_idx,1), ...
                    idx_mat(plot_idx,2), ...
                    idx_mat(plot_idx,3), ...
                    30, ...
                    cmap(hu_col_idx(plot_idx),:), ..., ...
                    'filled',...
                    "MarkerFaceAlpha",0.99, ...
                    "MarkerEdgeAlpha",0.99);
        view([0,1,0]);
        axis off; axis equal;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Scatter plot of convex hull indicator field
        figure;
        ch_pts=pts(:,ch_col);
        cmap=colormap;
        num_cols=size(cmap,1);
        ch_col_idx=(ch_pts-min(ch_pts))/(max(ch_pts)-min(ch_pts))*num_cols;
        ch_col_idx(ch_col_idx<1)=1;
        ch_col_idx(ch_col_idx>num_cols)=num_cols;
        ch_col_idx=round(ch_col_idx);
        ch_cols=cmap(ch_col_idx,:);
        y_min=min(idx_mat(:,2));
        y_max=max(idx_mat(:,2));
        y_range=y_max-y_min;
        plot_idx=find(idx_mat(:,2)<(y_min+0.50*(y_max-y_min)));
        plot_idx=plot_idx([1:10:numel(plot_idx)]);
        scatter3(idx_mat(plot_idx,1), ...
                    idx_mat(plot_idx,2), ...
                    idx_mat(plot_idx,3), ...
                    30, ...
                    cmap(ch_col_idx(plot_idx),:), ..., ...
                    'filled',...
                    "MarkerFaceAlpha",0.99, ...
                    "MarkerEdgeAlpha",0.99);
        view([0,1,0]);
        axis off; axis equal;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Create 3D image containing SDF values
        figure;
        sdf_pts=pts(:,sdf_col);
        cmap=colormap;
        num_cols=size(cmap,1);
        sdf_col_idx=(sdf_pts-min(sdf_pts))/(max(sdf_pts)-min(sdf_pts))*num_cols;
        sdf_col_idx(sdf_col_idx<1)=1;
        sdf_col_idx(sdf_col_idx>num_cols)=num_cols;
        sdf_col_idx=round(sdf_col_idx);
        sdf_cols=cmap(sdf_col_idx,:);
        temp_3d_im=zeros(x_dim,y_dim,z_dim,3);
        for k=1:128%in_z_idx:max_z_idx
            for j=1:128%min_y_idx:max_y_idx
                for i=1:128%min_x_idx:max_x_idx

                    v_idx=(k-1)*(x_dim*y_dim)+(j-1)*(x_dim)+(i);
                    cur_sdf_val=pts(v_idx,4);

                    temp_3d_im(i,j,k,:)=sdf_cols(v_idx,:);

                end
            end
        end
    
        imshow(squeeze(temp_3d_im(:,70,:,:)),[]);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Create 3D image containing intensity field
        hu_pts=pts(:,hu_col);
        col_min=[0,0,0];
        col_max=[0.9,0.9,0.99];
        num_cols=255;
        col_incs=[0:1:(num_cols-1)]'/(num_cols-1);
        cmap=col_min+col_incs.*(col_max-col_min);
        num_cols=size(cmap,1);
        hu_col_idx=(hu_pts-min(hu_pts))/(max(hu_pts)-min(hu_pts))*num_cols;
        hu_col_idx(hu_col_idx<1)=1;
        hu_col_idx(hu_col_idx>num_cols)=num_cols;
        hu_col_idx=round(hu_col_idx);
        hu_cols=cmap(hu_col_idx,:);
        
        temp_3d_im=zeros(x_dim,y_dim,z_dim,3);
        for k=1:128%in_z_idx:max_z_idx
            for j=1:128%min_y_idx:max_y_idx
                for i=1:128%min_x_idx:max_x_idx

                    v_idx=(k-1)*(x_dim*y_dim)+(j-1)*(x_dim)+(i);
                    temp_3d_im(i,j,k,:)=hu_cols(v_idx,:);

                end
            end
        end
    
        imshow(squeeze(temp_3d_im(:,70,:,:)),[]);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Create 3D image containing convex hull data
        ch_pts=pts(:,ch_col);
        cmap=colormap;
        num_cols=size(cmap,1);
        ch_col_idx=(ch_pts-min(ch_pts))/(max(ch_pts)-min(ch_pts))*num_cols;
        ch_col_idx(ch_col_idx<1)=1;
        ch_col_idx(ch_col_idx>num_cols)=num_cols;
        ch_col_idx=round(ch_col_idx);
        ch_cols=cmap(ch_col_idx,:);

        temp_3d_im=zeros(x_dim,y_dim,z_dim,3);
        for k=1:128%in_z_idx:max_z_idx
            for j=1:128%min_y_idx:max_y_idx
                for i=1:128%min_x_idx:max_x_idx
                    
                    v_idx=(k-1)*(x_dim*y_dim)+(j-1)*(x_dim)+(i);
                    
                    temp_3d_im(i,j,k,:)=ch_cols(v_idx,:);

                end
            end
        end
    
        imshow(squeeze(temp_3d_im(:,70,:,:)),[]);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Gaussian smoothing of SDF values
    if 1==1
        
        % Create temp 3D image containing SDF values
        temp_3d_im=zeros(x_dim,y_dim,z_dim);
        for k=min_z_idx:max_z_idx
            for j=min_y_idx:max_y_idx
                for i=min_x_idx:max_x_idx
                    v_idx=(k)*(x_dim*y_dim)+(j)*(x_dim)+(i);
                    cur_sdf_val=pts(v_idx,4);
                    temp_3d_im(i,j,k)=cur_sdf_val;
                end
            end
        end
        
        if 1==1
            for jj=1:10:size(temp_3d_im,2)
                imshow(squeeze(temp_3d_im(:,jj,:)),[]);
            end
        end
        
        % Apply 3D gussian filter
        sigma=2;
        volSmooth = imgaussfilt3(temp_3d_im, [1,1,0.5]);
        %volSmooth = medfilt3(temp_3d_im);

        pts_smoothed=pts;
        for k=min_z_idx:max_z_idx
            for j=min_y_idx:max_y_idx
                for i=min_x_idx:max_x_idx

                    v_idx=(k)*(x_dim*y_dim)+(j)*(x_dim)+(i);
                    cur_sdf_val=volSmooth(i,j,k); % Unpack value from smoothed 3d image
                    pts_smoothed(v_idx,4)=cur_sdf_val; % Store in original variable

                end
            end
        end

    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Apply marching points
    [n,e]=marching_points(x_dim, y_dim, z_dim, sdf_col, ...
                                min_x_idx, max_x_idx, ...
                                min_y_idx, max_y_idx, ...
                                min_z_idx, max_z_idx, ...
                                idx_mat, pts);
    [k]=boundary(n,0.99); % Use alpha shapes to wrap point cloud into closed surface mesh
    [n_surf,e_surf]=restructure_alpha_mesh(n,k);
    [n_surf,e_surf]=remove_outlier_elements(n_surf,e_surf);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Apply marching points to smoothed geom
    [n,e]=marching_points(x_dim, y_dim, z_dim, sdf_col, ...
                                min_x_idx, max_x_idx, ...
                                min_y_idx, max_y_idx, ...
                                min_z_idx, max_z_idx, ...
                                idx_mat, pts_smoothed);
    [k]=boundary(n,0.95);
    [n_surf_sm,e_surf_sm]=restructure_alpha_mesh(n,k);
    [n_surf_sm,e_surf_sm]=remove_outlier_elements(n_surf_sm,e_surf_sm);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
end



