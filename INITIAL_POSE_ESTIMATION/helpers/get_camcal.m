function [camcal] = get_camcal(path)

    % William Burton, University of Denver, 2025
    % Reads camera calibration data into a struct, camera calibration data is stored according to MayaCam2 format fro XMA-Lab (open source software for calibration biplanar radiography systems)

    inputfile = (path);
    fin = fopen(inputfile,'r');
    while ~feof(fin) %Do it until the end of the file

        temp=fgetl(fin);

        if length(temp)>2

            if strcmp(temp(1:3),'ima') == 1
                im = textscan(fin, '%f %f','Delimiter', ',');
            elseif strcmp(temp(1:3),'cam') == 1
                cm = textscan(fin, '%f %f %f','Delimiter', ',');
            elseif strcmp(temp(1:3),'rot') == 1
                r = textscan(fin, '%f %f %f','Delimiter', ',');
            elseif strcmp(temp(1:3),'tra') == 1
                t = textscan(fin, '%f','Delimiter', ',');
            end

        end

    end
    fclose(fin);

    camcal=struct();
    camcal.IM=[im{1}, im{2}];
    camcal.R=[r{1}, r{2}, r{3}];
    camcal.T=t{1};
    camcal.CM=[cm{1}, cm{2}, cm{3}];
    camcal.fx=camcal.CM(1,1);
    camcal.fy=camcal.CM(2,2);
    camcal.cx=camcal.CM(1,3);
    camcal.cy=camcal.CM(2,3);
    
    
    
end

