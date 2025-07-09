function [new_nodes, new_elems]=restructure_alpha_mesh(geom_pts, k)

     % William Burton, 2025, University of Denver
     % Given a list of nodes and elements that only capture a subset of those nodes, this function returns an updated mesh that contains only nodes which are attached to elements -- gets rid of unattached nodes
    
    % Get list of nodes that is present in element list
    filt_nodes_idx=[];
    filt_nodes_idx=[unique(k(:,1)); unique(k(:,2)); unique(k(:,3))];
    filt_nodes_idx=unique(filt_nodes_idx);

    new_nodes=[];
    new_elems=[];
    e_allocated=zeros(size(k));

    for i=1:length(filt_nodes_idx)
        
        old_node_idx=filt_nodes_idx(i);

        % Add node to pile of new nodes
        new_nodes=[new_nodes; geom_pts(old_node_idx,:)];
        
        % Update elements locations
        [row, col]=find(k==old_node_idx);
        for j=1:length(row(:,1))
           
           my_row=row(j);
           my_col=col(j);
           
           allocated_flag=e_allocated(my_row, my_col);
         
           if (allocated_flag==0)
               e_allocated(my_row, my_col)=1;
               new_elems(my_row, my_col)=i;
           end
           
        end

    end
    
    % Make sure all elements have been re-allocated
    e_unalloc=[
                find(e_allocated(:,1)==0); ...
                find(e_allocated(:,2)==0);...
                find(e_allocated(:,3)==0);
                ];
    e_unalloc=unique(e_unalloc);

    if min(size(e_unalloc))>0
        disp("WARNING: Unallocated elements");
    end

    


end
