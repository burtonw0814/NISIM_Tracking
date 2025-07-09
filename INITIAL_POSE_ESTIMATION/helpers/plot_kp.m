function [imd]=plot_kp(imd, kp, color_channel)

    % William Burton, University of Denver, 2025
    % Plot key points / projected nodes on image plane
    
    ht=size(imd,1);
    wd=size(imd,2);

    for j=1:size(kp,1)

        x_core=round(kp(j,1));
        y_core=round(kp(j,2));
        
        buf=2;
        for ii=-buf:buf
            for jj=-buf:buf
                
                x=x_core+ii;
                y=y_core+jj;
                
                if x<1
                    x=1;
                end

                if x>ht
                    x=ht;
                end

                if y<1
                    y=1;
                end

                if y>wd
                    y=wd;
                end

                imd(y,x,1)=0;
                imd(y,x,2)=0;
                imd(y,x,3)=0;
                imd(y,x,color_channel)=255;
                
            end
        end
        

    end


    
end
