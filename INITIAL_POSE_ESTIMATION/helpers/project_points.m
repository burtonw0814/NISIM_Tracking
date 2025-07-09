function [kp] = project_points(n, fx, fy, cx, cy, pix_wd, pix_ht)

    % Project nodes onto image plane using known intrinsics
        
    kp_x=n(:,1)./n(:,3).*fx+cx;
    kp_y=(n(:,2)./n(:,3).*fy+cy); %pix_ht-
    kp=[kp_x, kp_y];

    %kp_x_h=n(:,1)./n(:,3);
    %kp_y_h=n(:,2)./n(:,3);
    %kp_z_h=n(:,3)./n(:,3);
    %kp_h=[kp_x_h, kp_y_h, kp_z_h];
    
end
