function [n]=edge_interp(v0,v1)

    % William Burton, 2025, University of Denver
    % Find point along edge at which SDF value is 0

    x0=v0(1);
    y0=v0(2);
    z0=v0(3);
    sdf0=v0(4);
    
    x1=v1(1);
    y1=v1(2);
    z1=v1(3);
    sdf1=v1(4);
    
    if (sdf0>0 && sdf1>0) || (sdf0<0 && sdf1<0)
        disp("Warning: SDF issue!");
    end

    if sdf1>sdf0
        nx=(0-sdf0)/(sdf1-sdf0)*(x1-x0)+x0;
        ny=(0-sdf0)/(sdf1-sdf0)*(y1-y0)+y0;
        nz=(0-sdf0)/(sdf1-sdf0)*(z1-z0)+z0;
    else
        nx=(0-sdf1)/(sdf0-sdf1)*(x0-x1)+x1;
        ny=(0-sdf1)/(sdf0-sdf1)*(y0-y1)+y1;
        nz=(0-sdf1)/(sdf0-sdf1)*(z0-z1)+z1;
    end
    
    n=[nx,ny,nz];
    
end
