function [n,e]=marching_points(x_dim, y_dim, z_dim, sdf_col, min_x_idx, max_x_idx, min_y_idx, max_y_idx, min_z_idx, max_z_idx, idx_mat, pts)
    
    %[n,e]
    %[n_ct_check,e_ct_check]

    % Rebuild using naive "marching cubes-esque" approach
    % Specifically, create voxels from adjacent query points in 3D space, then keep all negative SDF values and interpolate voxel edges to find boundaries
    % Logic:
    
    n=[];
    e=[]; % TO-DO: Build elements
    n_ct=1;
    
    n_check=[];
    e_check=[];
    n_ct_check=1;

    n_cells=cell(max_z_idx,1);

    for k=min_z_idx:max_z_idx

        n=[];
        
        if mod(k,10)==0
            disp(['slice ' num2str(k) ' out of ' num2str(max_z_idx)]);
        end
        
        for j=min_y_idx:max_y_idx
            for i=min_x_idx:max_x_idx
                
                v0_idx=(k-1)*  (x_dim*y_dim)+(j-1)*(x_dim)+    (i);
                v1_idx=(k-1+1)*(x_dim*y_dim)+(j-1)*(x_dim)+    (i);
                v2_idx=(k-1)*  (x_dim*y_dim)+(j-1+1)*(x_dim)+  (i);
                v3_idx=(k-1+1)*(x_dim*y_dim)+(j-1+1)*(x_dim)+  (i);
    
                v4_idx=(k-1)*  (x_dim*y_dim)+(j-1)*(x_dim)+    (i+1);
                v5_idx=(k-1+1)*(x_dim*y_dim)+(j-1)*(x_dim)+    (i+1);
                v6_idx=(k-1)*  (x_dim*y_dim)+(j-1+1)*(x_dim)+  (i+1);
                v7_idx=(k-1+1)*(x_dim*y_dim)+(j-1+1)*(x_dim)+  (i+1);
                
                % i/j/k triplets for each voxel corner
                trip0=[i,  j,  k];
                trip1=[i,  j,  k+1];
                trip2=[i,  j+1,k];
                trip3=[i,  j+1,k+1];
    
                trip4=[i+1,j,  k];
                trip5=[i+1,j,  k+1];
                trip6=[i+1,j+1,k];
                trip7=[i+1,j+1,k+1];
                
                % Assert that indices line up
                idx_check0=idx_mat(v0_idx,:);
                idx_check1=idx_mat(v1_idx,:);
                idx_check2=idx_mat(v2_idx,:);
                idx_check3=idx_mat(v3_idx,:);
    
                idx_check4=idx_mat(v4_idx,:);
                idx_check5=idx_mat(v5_idx,:);
                idx_check6=idx_mat(v6_idx,:);
                idx_check7=idx_mat(v7_idx,:);
    
                all_trips=[trip0; trip1; trip2; trip3;
                           trip4; trip5; trip6; trip7];
                all_idx_checks=[idx_check0; idx_check1; 
                                idx_check2; idx_check3;
                                idx_check4; idx_check5; 
                                idx_check6; idx_check7;
                                ];
                recon_difs=sum(sum(abs(all_trips-all_idx_checks)));
                if recon_difs>0
                    disp("Warning reconstrution failed");
                end
                
                v0=pts(v0_idx,:);
                v1=pts(v1_idx,:);
                v2=pts(v2_idx,:);
                v3=pts(v3_idx,:);
                v4=pts(v4_idx,:);
                v5=pts(v5_idx,:);
                v6=pts(v6_idx,:);
                v7=pts(v7_idx,:);
                
    
                % Determine how many corners in this cell are negative
                seg_ct=0;
                neg_id=zeros(8,1);
                if v0(sdf_col)<0
                    seg_ct=seg_ct+1; neg_id(1)=1;
                end
                if v1(sdf_col)<0
                    seg_ct=seg_ct+1; neg_id(2)=1;
                end
                if v2(sdf_col)<0
                    seg_ct=seg_ct+1; neg_id(3)=1;
                end
                if v3(sdf_col)<0
                    seg_ct=seg_ct+1; neg_id(4)=1;
                end
                if v4(sdf_col)<0
                    seg_ct=seg_ct+1; neg_id(5)=1;
                end
                if v5(sdf_col)<0
                    seg_ct=seg_ct+1; neg_id(6)=1;
                end
                if v6(sdf_col)<0
                    seg_ct=seg_ct+1; neg_id(7)=1;
                end
                if v7(sdf_col)<0
                    seg_ct=seg_ct+1; neg_id(8)=1;
                end


                

                if seg_ct>0 % If any corners have negative SDF values

                    if seg_ct==8 % Store all corners for cells with all corners negative!
                        %n=[n; v0(1:3); v1(1:3); v2(1:3); v3(1:3); v4(1:3); v5(1:3); v6(1:3); v7(1:3); ];
                        new_n=(v0(1:3)+v1(1:3)+v2(1:3)+v3(1:3)+v4(1:3)+v5(1:3)+v6(1:3)+v7(1:3))/8;
                        n=[n; new_n];

                    else % Otherwise interpolate node positions

                        % Check all edges of cubes
                        if abs(neg_id(1)-neg_id(2))>0 %0,1 % Left face
                            new_n=edge_interp(v0(1:4),v1(1:4));
                            n=[n; new_n];
                        elseif abs(neg_id(2)-neg_id(4))>0 %1,3
                            new_n=edge_interp(v1(1:4),v3(1:4));
                            n=[n; new_n];
                        elseif abs(neg_id(3)-neg_id(4))>0 %2,3
                            new_n=edge_interp(v2(1:4),v3(1:4));
                            n=[n; new_n];
                        elseif abs(neg_id(3)-neg_id(1))>0 %2,0
                            new_n=edge_interp(v2(1:4),v0(1:4));
                            n=[n; new_n];

                        elseif abs(neg_id(5)-neg_id(7))>0 %4,6 % Right face
                            new_n=edge_interp(v4(1:4),v6(1:4));
                            n=[n; new_n];
                        elseif abs(neg_id(5)-neg_id(6))>0 %4,5 
                            new_n=edge_interp(v4(1:4),v5(1:4));
                            n=[n; new_n];
                        elseif abs(neg_id(6)-neg_id(8))>0 %5,7
                            new_n=edge_interp(v5(1:4),v7(1:4));
                            n=[n; new_n];
                        elseif abs(neg_id(7)-neg_id(8))>0 %6,7 
                            new_n=edge_interp(v6(1:4),v7(1:4));
                            n=[n; new_n];
                            
                        elseif abs(neg_id(3)-neg_id(7))>0 %2,6 % Remaining front face
                            new_n=edge_interp(v2(1:4),v6(1:4));
                            n=[n; new_n];
                        elseif abs(neg_id(1)-neg_id(5))>0 %0,4
                            new_n=edge_interp(v0(1:4),v4(1:4));
                            n=[n; new_n];

                        elseif abs(neg_id(4)-neg_id(8))>0 %3,7 % Remaining back face
                            new_n=edge_interp(v3(1:4),v7(1:4));
                            n=[n; new_n];
                        elseif abs(neg_id(2)-neg_id(6))>0 %1,5
                            new_n=edge_interp(v1(1:4),v5(1:4));
                            n=[n; new_n];
                            
                        else
                            disp("Missing edge condition")
                        end
                        
                    end


                end
                
            end % End x iters
        end % End y iters
        
        n_cells{k}=n;
        
    end % end z iters
    
    % TO-DO: Iterate over all, for any positive, get surrounding, if all surrounding are negative, add as negative (could potentiall mitigate internal cavities)
    
    n=[];
    for ii=1:numel(n_cells)
        n=[n; n_cells{ii}];
    end

    disp("Done");

end
