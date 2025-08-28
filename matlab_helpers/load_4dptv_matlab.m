clear, clc

s = load('calib.mat');

[nRows, nCols] = size(s.calib);

for i = 1:nRows
    for j = 1:nCols
        c = s.calib{i,j};   % get struct in this cell
        
        A_rw2px = c.T3rw2px(1,:);
        B_rw2px = c.T3rw2px(2,:);
        A_px2rw = c.T3px2rw(1,:);
        B_px2rw = c.T3px2rw(2,:);
        
        c.T3rw2px = images.geotrans.PolynomialTransformation2D(A_rw2px, B_rw2px);
        c.T3px2rw = images.geotrans.PolynomialTransformation2D(A_px2rw, B_px2rw);
        
        s.calib{i,j} = c;   % put back the updated struct
    end
end