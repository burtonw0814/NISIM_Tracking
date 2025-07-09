function [Rp, Rp2, Tp]=opt1(n, kp0, kp1, camcal0, camcal1)

    % William Burton, University of Denver, 2025
    % Biplanar PnP solved with semidefinite relaxation approach (because it is non-convex)
    % See Burton et al 2024, including Supplementary Material
    
    % n=cur_nodes
    num_nodes=size(n,1);

    % Get homogenous form of key points
    kph0=[kp0, ones(size(kp0,1),1)];
    kph0(:,1)=(kph0(:,1)-camcal0.cx)/camcal0.fx;
    kph0(:,2)=(kph0(:,2)-camcal0.cy)/camcal0.fy; %camcal0.IM(1)-

    kph1=[kp1, ones(size(kp1,1),1)];
    kph1(:,1)=(kph1(:,1)-camcal1.cx)/camcal1.fx;
    kph1(:,2)=(kph1(:,2)-camcal1.cy)/camcal1.fy;% camcal1.IM(1)-
    
    % Store all P vals and V' vals for CAM 1 and 2
    P=cell(num_nodes,1);
    Vp0=cell(num_nodes,1);
    Vp1=cell(num_nodes,1);
    for i=1:num_nodes
        P{i}=double(n(i,:)');
        
        vi=double(kph0(i,:)'); % Get homogenous coordinates
        Vi=(vi*vi')./(vi'*vi); % Get projection matrix
        Vp0{i}=eye(3)-Vi; % Get diff
        
        vi=double(kph1(i,:)'); % Get homogenous coordinates
        Vi=(vi*vi')./(vi'*vi); % Get projection matrix
        Vp1{i}=eye(3)-Vi; % Get diff
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Get objective function data
    % Get K
    K_inv=zeros(3,3);
    for i=1:num_nodes
    
        % Unpack data
        Rc0=camcal0.R;
        Tc0=camcal0.T;
        
        Rc1=camcal1.R;
        Tc1=camcal1.T;
    
        V0=Vp0{i};
        V1=Vp1{i};
    
        A=P{i};
    
        % Accumulate intermediate variables
        K_inv=K_inv+(V0*Rc0)'*(V0*Rc0);
        K_inv=K_inv+(V1*Rc1)'*(V1*Rc1);
        
    end
    K=inv(K_inv);


    
    % Construct C, D
    D=zeros(3,9);
    C=zeros(3,1);
    for i=1:num_nodes
        
        % Unpack data
        Rc0=camcal0.R;
        Tc0=camcal0.T;
        
        Rc1=camcal1.R;
        Tc1=camcal1.T;
    
        V0=Vp0{i};
        V1=Vp1{i};
        
        A=P{i};
        B1=(V0*Rc0)'*(V0*Rc0);
        B2=(V1*Rc1)'*(V1*Rc1);
    
        new_D1=[A(1)*B1(1,1), A(1)*B1(1,2) A(1)*B1(1,3) , ...
                A(2)*B1(1,1), A(2)*B1(1,2) A(2)*B1(1,3) , ...
                A(3)*B1(1,1), A(3)*B1(1,2) A(3)*B1(1,3) ; ...
                ...
                A(1)*B1(2,1), A(1)*B1(2,2) A(1)*B1(2,3) , ...
                A(2)*B1(2,1), A(2)*B1(2,2) A(2)*B1(2,3) , ...
                A(3)*B1(2,1), A(3)*B1(2,2) A(3)*B1(2,3) ; ...
                ...
                A(1)*B1(3,1), A(1)*B1(3,2) A(1)*B1(3,3) , ...
                A(2)*B1(3,1), A(2)*B1(3,2) A(2)*B1(3,3) , ...
                A(3)*B1(3,1), A(3)*B1(3,2) A(3)*B1(3,3) ; ...
                ];
        
        new_D2=[A(1)*B2(1,1), A(1)*B2(1,2) A(1)*B2(1,3) , ...
                A(2)*B2(1,1), A(2)*B2(1,2) A(2)*B2(1,3) , ...
                A(3)*B2(1,1), A(3)*B2(1,2) A(3)*B2(1,3) ; ...
                ...
                A(1)*B2(2,1), A(1)*B2(2,2) A(1)*B2(2,3) , ...
                A(2)*B2(2,1), A(2)*B2(2,2) A(2)*B2(2,3) , ...
                A(3)*B2(2,1), A(3)*B2(2,2) A(3)*B2(2,3) ; ...
                ...
                A(1)*B2(3,1), A(1)*B2(3,2) A(1)*B2(3,3) , ...
                A(2)*B2(3,1), A(2)*B2(3,2) A(2)*B2(3,3) , ...
                A(3)*B2(3,1), A(3)*B2(3,2) A(3)*B2(3,3) ; ...
                ];
             
        D=D+new_D1+new_D2; % 
        
        C=C+(-1)*K*(V0*Rc0)'*(V0*Tc0);
        C=C+(-1)*K*(V1*Rc1)'*(V1*Tc1);
    end
    
    Q=zeros(10,10);
    for i=1:num_nodes
       
       % Unpack data
       Rc0=camcal0.R;
       Tc0=camcal0.T;
        
       Rc1=camcal1.R;
       Tc1=camcal1.T;
    
       V0=Vp0{i};
       V1=Vp1{i};
    
       A=P{i};
       
       % 
       B1=(V0*Rc0);
       B2=(V1*Rc1);
    
       % Construct F
       F1=[A(1)*B1(1,1), A(1)*B1(1,2) A(1)*B1(1,3) , ...
           A(2)*B1(1,1), A(2)*B1(1,2) A(2)*B1(1,3) , ...
           A(3)*B1(1,1), A(3)*B1(1,2) A(3)*B1(1,3) ; ...
           ...
           A(1)*B1(2,1), A(1)*B1(2,2) A(1)*B1(2,3) , ...
           A(2)*B1(2,1), A(2)*B1(2,2) A(2)*B1(2,3) , ...
           A(3)*B1(2,1), A(3)*B1(2,2) A(3)*B1(2,3) ; ...
           ...
           A(1)*B1(3,1), A(1)*B1(3,2) A(1)*B1(3,3) , ...
           A(2)*B1(3,1), A(2)*B1(3,2) A(2)*B1(3,3) , ...
           A(3)*B1(3,1), A(3)*B1(3,2) A(3)*B1(3,3) ; ...
           ];
            
        F2=[A(1)*B2(1,1), A(1)*B2(1,2) A(1)*B2(1,3) , ...
            A(2)*B2(1,1), A(2)*B2(1,2) A(2)*B2(1,3) , ...
            A(3)*B2(1,1), A(3)*B2(1,2) A(3)*B2(1,3) ; ...
            ...
            A(1)*B2(2,1), A(1)*B2(2,2) A(1)*B2(2,3) , ...
            A(2)*B2(2,1), A(2)*B2(2,2) A(2)*B2(2,3) , ...
            A(3)*B2(2,1), A(3)*B2(2,2) A(3)*B2(2,3) ; ...
            ...
            A(1)*B2(3,1), A(1)*B2(3,2) A(1)*B2(3,3) , ...
            A(2)*B2(3,1), A(2)*B2(3,2) A(2)*B2(3,3) , ...
            A(3)*B2(3,1), A(3)*B2(3,2) A(3)*B2(3,3) ; ...
            ];
        
        % Construct G
        G1=V0*Rc0*(-1)*K*D;
        G2=V1*Rc1*(-1)*K*D;
        
        % Construct H
        H1=V0*Rc0*C+V0*Tc0;
        H2=V1*Rc1*C+V1*Tc1;
        
        % J from F, G
        J1=F1+G1;
        J2=F2+G2;
        
        % Qp from J, H
        Qp1=[H1,J1];
        Qp2=[H2,J2];
        
        % New Q from Qp
        Q1=Qp1'*Qp1;
        Q2=Qp2'*Qp2;
        
        % 
        Q=Q+Q1;
        Q=Q+Q2;
        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Get constraints data
    
    % Store monomial degrees for each element in design matrix
    monomial_cells=cell(10,10);
    for ii=1:size(monomial_cells,1)
        for jj=1:size(monomial_cells,2)
            
            % Determine power for this cell
            px=jj;
            py=ii;
            my_p=zeros(10,1);
            my_p(px)=my_p(px)+1;
            my_p(py)=my_p(py)+1;
            
            % Fix zero-th degree issues
            if ii==1 && jj==1
                my_p(1)=1;
            else
                my_p(1)=0;
            end
    
            % Store it
            monomial_cells{ii,jj}=my_p;
            
        end
    end
    
    % TODO: Explore different data structures?
    
    % Create list of constraints
    monomials=cell(15,1); % Monomial terms in each constraint polynomial
    coeffs=cell(15,1); % Coefficients associated with each monomials in each constraint
                
                %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{1}=[  1   0   0   0   0   0   0   0   0   0; ... 
                    0   2   0   0   0   0   0   0   0   0; ...
                    0   0   2   0   0   0   0   0   0   0; ...
                    0   0   0   2   0   0   0   0   0   0; ];
    coeffs{1}=[-1, 1, 1, 1]; % Each coeff corresponds to monomial defined in each power row
    
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{2}=[  1   0   0   0   0   0   0   0   0   0; ... 
                    0   0   0   0   2   0   0   0   0   0; ...
                    0   0   0   0   0   2   0   0   0   0; ...
                    0   0   0   0   0   0   2   0   0   0;  ];
    coeffs{2}=[-1, 1, 1, 1];
    
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{3}=[  1   0   0   0   0   0   0   0   0   0; ... 
                    0   0   0   0   0   0   0   2   0   0; ...
                    0   0   0   0   0   0   0   0   2   0; ...
                    0   0   0   0   0   0   0   0   0   2;  ];
    coeffs{3}=[-1, 1, 1, 1];
    
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{4}=[  0   1   0   0   1   0   0   0   0   0; ... 
                    0   0   1   0   0   1   0   0   0   0; ...
                    0   0   0   1   0   0   1   0   0   0;];
    coeffs{4}=[1, 1, 1];
    
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{5}=[  0   0   0   0   1   0   0   1   0   0; ... 
                    0   0   0   0   0   1   0   0   1   0; ...
                    0   0   0   0   0   0   1   0   0   1;];
    coeffs{5}=[1, 1, 1];
    
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{6}=[  0   1   0   0   0   0   0   1   0   0; ... 
                    0   0   1   0   0   0   0   0   1   0; ...
                    0   0   0   1   0   0   0   0   0   1;];
    coeffs{6}=[1, 1, 1];
             
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{7}=[  0   0   1   0   0   0   1   0   0   0; ... 
                    0   0   0   1   0   1   0   0   0   0; ...
                    0   0   0   0   0   0   0   1   0   0;];
    coeffs{7}=[1, -1, -1];
    
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{8}=[  0   0   0   1   1   0   0   0   0   0; ... 
                    0   1   0   0   0   0   1   0   0   0; ...
                    0   0   0   0   0   0   0   0   1   0;];
    coeffs{8}=[1, -1, -1];
              
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{9}=[  0   1   0   0   0   1   0   0   0   0; ... 
                    0   0   1   0   1   0   0   0   0   0; ...
                    0   0   0   0   0   0   0   0   0   1;];
    coeffs{9}=[1, -1, -1];
            
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{10}=[ 0   0   0   0   0   1   0   0   0   1; ... 
                    0   0   0   0   0   0   1   0   1   0; ...
                    0   1   0   0   0   0   0   0   0   0;];
    coeffs{10}=[1, -1, -1];
    
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{11}=[ 0   0   0   0   0   0   1   1   0   0; ... 
                    0   0   0   0   1   0   0   0   0   1; ...
                    0   0   1   0   0   0   0   0   0   0;];
    coeffs{11}=[1, -1, -1];
    
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{12}=[ 0   0   0   0   1   0   0   0   1   0; ... 
                    0   0   0   0   0   1   0   1   0   0; ...
                    0   0   0   1   0   0   0   0   0   0;];
    coeffs{12}=[1, -1, -1];
    
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{13}=[ 0   0   0   1   0   0   0   0   1   0; ... 
                    0   0   1   0   0   0   0   0   0   1; ...
                    0   0   0   0   1   0   0   0   0   0;];
    coeffs{13}=[1, -1, -1];
    
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{14}=[ 0   1   0   0   0   0   0   0   0   1; ... 
                    0   0   0   1   0   0   0   1   0   0; ...
                    0   0   0   0   0   1   0   0   0   0;];
    coeffs{14}=[1, -1, -1];
            
                 %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{15}=[ 0   0   1   0   0   0   0   1   0   0; ... 
                    0   1   0   0   0   0   0   0   1   0; ...
                    0   0   0   0   0   0   1   0   0   0;];
    coeffs{15}=[1, -1, -1];
    
    
    
    
    % Redundant constraints from R rows instead of cols
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{16}=[  1   0   0   0   0   0   0   0   0   0; ... 
                     0   2   0   0   0   0   0   0   0   0; ...
                     0   0   0   0   2   0   0   0   0   0; ...
                     0   0   0   0   0   0   0   2   0   0; ];
    coeffs{16}=[-1, 1, 1, 1]; % Each coeff corresponds to monomial defined in each power row
                    
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{17}=[  1   0   0   0   0   0   0   0   0   0; ... 
                     0   0   2   0   0   0   0   0   0   0; ...
                     0   0   0   0   0   2   0   0   0   0; ...
                     0   0   0   0   0   0   0   0   2   0; ];
    coeffs{17}=[-1, 1, 1, 1]; % Each coeff corresponds to monomial defined in each power row
                    
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{18}=[  1   0   0   0   0   0   0   0   0   0; ... 
                     0   0   0   2   0   0   0   0   0   0; ...
                     0   0   0   0   0   0   2   0   0   0; ...
                     0   0   0   0   0   0   0   0   0   2 ; ];
    coeffs{18}=[-1, 1, 1, 1]; % Each coeff corresponds to monomial defined in each power row
    
    
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{19}=[  0   1   1   0   0   0   0   0   0   0; ... 
                     0   0   0   0   1   1   0   0   0   0; ...
                     0   0   0   0   0   0   0   1   1   0;  ];
    coeffs{19}=[1, 1, 1]; % Each coeff corresponds to monomial defined in each power row
    
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{20}=[  0   0   1   1   0   0   0   0   0   0; ... 
                     0   0   0   0   0   1   1   0   0   0; ...
                     0   0   0   0   0   0   0   0   1   1;];
    coeffs{20}=[1, 1, 1]; % Each coeff corresponds to monomial defined in each power row
    
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{21}=[  0   1   0   1   0   0   0   0   0   0; ... 
                     0   0   0   0   1   0   1   0   0   0; ...
                     0   0   0   0   0   0   0   1   0   1; ];
    coeffs{21}=[1, 1, 1]; % Each coeff corresponds to monomial defined in each power row
    
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{22}=[  0   0   0   0   1   0   0   0   1   0; ... 
                     0   0   0   0   0   1   0   1   0   0; ...
                     0   0   0   1   0   0   0   0   0   0; ];
    coeffs{22}=[1, -1, -1]; % Each coeff corresponds to monomial defined in each power row
    
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{23}=[  0   0   1   0   0   0   0   1   0   0; ...
                     0   1   0   0   0   0   0   0   1   0; ... 
                     0   0   0   0   0   0   1   0   0   0; ];
    coeffs{23}=[1, -1, -1]; % Each coeff corresponds to monomial defined in each power row
    
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{24}=[  0   1   0   0   0   1   0   0   0   0; ... 
                     0   0   1   0   1   0   0   0   0   0; ...
                     0   0   0   0   0   0   0   0   0   1; ];
    coeffs{24}=[1, -1, -1]; % Each coeff corresponds to monomial defined in each power row
    
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{25}=[  0   0   0   0   0   1   0   0   0   1; ... 
                     0   0   0   0   0   0   1   0   1   0; ...
                     0   1   0   0   0   0   0   0   0   0; ];
    coeffs{25}=[1, -1, -1]; % Each coeff corresponds to monomial defined in each power row
    
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{26}=[  0   0   0   1   0   0   0   0   1   0; ...
                     0   0   1   0   0   0   0   0   0   1; ... 
                     0   0   0   0   1   0   0   0   0   0; ];
    coeffs{26}=[1, -1, -1]; % Each coeff corresponds to monomial defined in each power row
                
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{27}=[  0   0   1   0   0   0   1   0   0   0; ... 
                     0   0   0   1   0   1   0   0   0   0; ...
                     0   0   0   0   0   0   0   1   0   0; ];
    coeffs{27}=[1, -1, -1]; % Each coeff corresponds to monomial defined in each power row
                  
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{28}=[  0   0   0   0   0   0   1   1   0   0; ... 
                     0   0   0   0   1   0   0   0   0   1; ...
                     0   0   1   0   0   0   0   0   0   0; ];
    coeffs{28}=[1, -1, -1]; % Each coeff corresponds to monomial defined in each power row
    
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{29}=[  0   1   0   0   0   0   0   0   0   1; ...
                     0   0   0   1   0   0   0   1   0   0; ... 
                     0   0   0   0   0   1   0   0   0   0; ];
    coeffs{29}=[1, -1, -1]; % Each coeff corresponds to monomial defined in each power row
    
                  %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    monomials{30}=[  0   0   0   1   1   0   0   0   0   0; ... 
                     0   1   0   0   0   0   1   0   0   0; ...
                     0   0   0   0   0   0   0   0   1   0; ];
    coeffs{30}=[1, -1, -1]; % Each coeff corresponds to monomial defined in each power row
    
    
    
    %               %  1  r00 r10 r20 r01 r11 r21 r02 r12 r22
    % monomials{}=[  0   0   0   0   0   0   0   0   0   0; ... 
    %                  0   0   0   0   0   0   0   0   0   0; ...
    %                  0   0   0   0   0   0   0   0   0   0; ];
    % coeffs{}=[1, -1, -1]; % Each coeff corresponds to monomial defined in each power row
    % 
    
    
    
    
    % Define symmetric matrices from constraints for SDP format
    Qc=cell(numel(monomials)+1,1); % Symmetric matrix for constraint
    b=zeros(numel(monomials)+1,1); % Right hand side scalar of constraint
    for ii=1:numel(monomials) % 15 rotation constraints + redundants
        my_Q=zeros(10,10);
    
        my_mono=monomials{ii};
        my_coeffs=coeffs{ii};
        
        % Iterate over each monomial and find matching locations in monomial
        % matrix
        for jj=1:numel(my_coeffs) % All monomials in this constraint
            cur_mono=my_mono(jj,:);
            cur_coeff=my_coeffs(jj);
            idx=[];
            for kk=1:size(monomial_cells,1) % Rows of monomial matrix
                for ll=1:size(monomial_cells,2) % Cols of monomial matrix
                    if sum(abs(cur_mono-monomial_cells{kk,ll}'))==0 % Monomial matches
                        idx=[idx; [kk,ll]];
                    end
                end
            end
            num_els=size(idx,1);
            actual_coeff=cur_coeff/num_els; % Split total coeff
            for kk=1:num_els
                my_Q(idx(kk,1),idx(kk,2))=actual_coeff;
            end
        end
    
        % Store matrix
        Qc{ii}=my_Q;
    end
    
    % Unity constraint
    my_Q=zeros(10,10);
    my_Q(1,1)=1;
    Qc{end}=my_Q;
    b(end)=1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Define objective function
    yalmip('clear');
    sdpvar X(10,10);
    J=trace(Q'*X);

    % Constraints
    F=[X>=0;]; % PSD constraint
    for ii=1:numel(Qc)
        F=[F; ...
           trace(Qc{ii}'*X)==b(ii);];
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Solve semidef relaxation
    ops = sdpsettings('solver','sedumi','sedumi.eps',0); % was 1e-9. 5e-7 'verbose',0
    optimize(F,J);
   
    % Check rank at different tolerances to verify rank 1
    X_val=value(X);
    tol=0.0000001;
    rank(X_val,tol); % Check rank at different tolerances
    
    % Pull first column to get rotation
    x_vec=X_val(:,1);
    Rp=reshape(x_vec(2:end),3,3);
    
    [V,D]=eig(X_val);
    [d,ind] = sort(diag(D),"descend");
    Ds = D(ind,ind);
    Vs = V(:,ind);
    Vsn=Vs./Vs(1,:);

    [u,s,v]=svd(X_val);

    % https://math.stackexchange.com/questions/1927845/is-u-v-in-the-svd-of-a-symmetric-positive-semidefinite-matrix
    %if A is real symmetric and positive definite (i.e. all of its eigenvalues are strictly positive), Σ is a diagonal matrix containing the eigenvalues, and U=V.
    
    % Choose rank-1 decomp to match original constraint set!

    % however the signs in the left and right singular vectors can be interchanged. 
    % Also, remember that they are unit vectors: so they are either equal to vectors in Q or to −1 times these vectors.
    
    % if A is real symmetric then (spectral theorem) it is diagonalizable and therefore has at least one eigendecomposition 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Define new optimization problem for projection to original constraint set
    x = optimvar('x',10,1);
    obj=trace((x*x'-X_val)'*(x*x'-X_val)); % Frobenius norm squared of difference
    prob = optimproblem('Objective',obj);
    
    % Constraints
    c1=1-x(2)*x(2)-x(3)*x(3)-x(4)*x(4)==0;
    prob.Constraints.c1=c1;
    c2=1-x(5)*x(5)-x(6)*x(6)-x(7)*x(7)==0;
    prob.Constraints.c2=c2;
    c3=1-x(8)*x(8)-x(9)*x(9)-x(10)*x(10)==0;
    prob.Constraints.c3=c3;
    
    c4=x(2)*x(5)+x(3)*x(6)+x(4)*x(7)==0;
    prob.Constraints.c4=c4;
    c5=x(5)*x(8)+x(6)*x(9)+x(7)*x(10)==0;
    prob.Constraints.c5=c5;
    c6=x(2)*x(8)+x(3)*x(9)+x(4)*x(10)==0;
    prob.Constraints.c6=c6;
    
    c7=x(3)*x(7)-x(4)*x(6)-x(8)==0;
    prob.Constraints.c7=c7;
    c8=x(4)*x(5)-x(2)*x(7)-x(9)==0;
    prob.Constraints.c8=c8;
    c9=x(2)*x(6)-x(3)*x(5)-x(10)==0;
    prob.Constraints.c9=c9;
    
    c10=x(6)*x(10)-x(7)*x(9)-x(2)==0;
    prob.Constraints.c10=c10;
    c11=x(7)*x(8)-x(5)*x(10)-x(3)==0;
    prob.Constraints.c11=c11;
    c12=x(5)*x(9)-x(6)*x(8)-x(4)==0;
    prob.Constraints.c12=c12;
    
    c13=x(4)*x(9)-x(3)*x(10)-x(5)==0;
    prob.Constraints.c13=c13;
    c14=x(2)*x(10)-x(4)*x(8)-x(6)==0;
    prob.Constraints.c14=c14;
    c15=x(3)*x(8)-x(2)*x(9)-x(7)==0;
    prob.Constraints.c15=c15;
    
    c16=x(1)-1==0;
    prob.Constraints.c16=c16;
    show(prob);
    x0.x = X_val(:,1);
    options = optimoptions('fmincon','Display','none','Algorithm','sqp');
    [sol,fval,exitflag,output] = solve(prob,x0,"Options",options);
    trace((sol.x*sol.x'-X_val)'*(sol.x*sol.x'-X_val)); % Final residual of PSD matrix
    vecnorm(x0.x-sol.x); % Residual of vector
    Rp2=reshape(sol.x(2:end),3,3);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Now get T
    K_inv=zeros(3,3);
    D=zeros(3,1);
    C=zeros(3,1);
    
    for i=1:num_nodes
        
        % Unpack data
        Rc0=camcal0.R;
        Tc0=camcal0.T;
        
        Rc1=camcal1.R;
        Tc1=camcal1.T;
        
        V0=Vp0{i};
        V1=Vp1{i};
        
        A=P{i};
        
        % Accumulate intermediate variables
        K_inv=K_inv+(V0*Rc0)'*(V0*Rc0);
        K_inv=K_inv+(V1*Rc1)'*(V1*Rc1);
    
        B1=(V0*Rc0)'*(V0*Rc0);
        B2=(V1*Rc1)'*(V1*Rc1);
    
        D=D+B1*Rp*A;
        D=D+B2*Rp*A;
        
        C=C+(V0*Rc0)'*(V0*Tc0);
        C=C+(V1*Rc1)'*(V1*Tc1);
        
    end
    K=inv(K_inv);
    Tp=-1*K*(D+C);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    disp("Rank of semidefinite relaxation " + num2str(rank(X_val,tol)));
    disp("Projection norm " + num2str(trace((sol.x*sol.x'-X_val)'*(sol.x*sol.x'-X_val))));
    
end







