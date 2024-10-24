function [psnr,inc,maxpayload] = cnn_1D(jpgpath,npypath,payload,msg,QF)
jpgObj = jpeg_read(jpgpath);
jpgCoef = jpgObj.coef_arrays{1};
qtable = jpgObj.quant_tables{1};
[m,n] = size(jpgCoef);

predictionCoef = readNPY(npypath);

coef_efficiency = [];
for i = 1:m
    for j = 1:n
        if (mod(i,8) ~= 1 || mod(j,8) ~= 1) && jpgCoef(i,j) ~= 0
            pc = predictionCoef(i,j);
            ps = 1 - pc;
            L = pc;
            D = (pc/2 + ps) * qtable(mod(i-1,8)+1,mod(j-1,8)+1) ^ 2;
            coef_efficiency = [coef_efficiency,[L/D;i;j]];
        end
    end
end
coef_efficiency = coef_efficiency';
coef_ranking = sortrows(coef_efficiency, -1);
[maxc,~] = size(coef_ranking);

count = 1;
c = 1;
while count <= payload
    i = coef_ranking(c, 2);
    j = coef_ranking(c, 3);
    if abs(jpgCoef(i,j)) == 1
        jpgCoef(i,j) = jpgCoef(i,j) + sign(jpgCoef(i,j)) * msg(count);
        count = count + 1;
    else
        jpgCoef(i,j) = jpgCoef(i,j) + sign(jpgCoef(i,j));
    end
    c = c + 1;
    if c > maxc
        break;
    end
end

maxpayload = count;
if maxpayload <= payload
    psnr = 0;
    inc = 0;
else
    
    load(strcat('default_gray_jpeg_obj_', num2str(QF), '.mat'));
    stegoObj = default_gray_jpeg_obj;
    stegoObj.coef_arrays{1} = jpgCoef;
    stegoObj.optimize_coding = 1;
    stegoJPEG = 'stego.jpg';
    jpeg_write(stegoObj,stegoJPEG);
    temp = dir(stegoJPEG);
    fileSize = temp.bytes;
    temp2 = dir(jpgpath);
    inc = (temp.bytes-temp2.bytes)*8;
    
    cover = imread(jpgpath);
    stego = imread(stegoJPEG);
    diff = double(cover) - double(stego);
    x = sum(sum(diff.*diff));
    psnr= 10*log10(512*512*255*255/x);
end

end