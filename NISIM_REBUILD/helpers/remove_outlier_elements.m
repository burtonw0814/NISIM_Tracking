function [nso,eso]=remove_outlier_elements(nso,eso)

    % William Burton, 2025, University of Denver
    % Expresses triangular surface mesh as undirected graph, then uses standard connected components algorithm to retrieve max connected component and remove "rogue" element clusters that aren't part of the main
    
    % Create graph
    all_edges=[[eso(:,1), eso(:,2)]; [eso(:,2), eso(:,3)]; [eso(:,1), eso(:,3)]; ];
    my_graph=graph(all_edges(:,1), all_edges(:,2));
    [bins,bin_sizes]=conncomp(my_graph);
    if numel(unique(bins))>1

        % Get idx to max comp
        max_comp_idx=find(bin_sizes==max(bin_sizes));
        
        % Create new mesh from max comp
        new_n=zeros(bin_sizes(max_comp_idx),3);
        new_e=eso;
        e_alloc=zeros(size(eso));
        n_ct=1;
        for ii=1:size(nso,1) % Iterate over all nodes
            if bins(ii)==max_comp_idx
                new_n(n_ct,:)=nso(ii,:);
                
                % Find locations in elements
                [idx_y,idx_x]=find(new_e==ii);
                for kk=1:numel(idx_y)
                    if e_alloc(idx_y(kk),idx_x(kk))==0 % Hasnt been reassigned yet
                        new_e(idx_y(kk),idx_x(kk))=n_ct; % Update node id
                        e_alloc(idx_y(kk),idx_x(kk))=1; % Make sure we know we've already updated this element
                    end
                end
                n_ct=n_ct+1;
            end
        end 

        % Remove elements that weren't updated because they belong
        % to the outlier components
        idx_empty=find(sum(e_alloc,2)==0);
        new_e(idx_empty,:)=[];
        %idx_full=find(sum(e_alloc,2)==3);

        nso=new_n;
        eso=new_e;

    end % End size bins check
    
end
