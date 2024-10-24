function [EC]=countPayload(img)

Th = 1;
jpegOrder = [ 1 9 2 3 10 17 25 18   11 4 5 12 19 26 33 41   34 27 20 13 6 7 14 21   28 35 42 49 57 50 43 36,...
             29 22 15 8 16 23 30 37   44 51 58 59 52 45 38 31   24 32 39 46 53 60 61 54    47 40 48 55 62 63 56 64];   
jpgObj = jpeg_read(img); %read the original JPEG image
jpgCoef = jpgObj.coef_arrays{1}; %read the coefficients of JPEG image

jpgVecCoef = im2vec(jpgCoef,[8,8]);%通过im2vec得到jpgCoef矩阵的列向量阵列       
%按照jpegOrder的顺序将jpgVecCoef矩阵复制为jpgEmbCoef
jpgEmbCoef = jpgVecCoef(jpegOrder(1:64),:);  % the 1:64 frequency bands are selected for embeding

S = jpgEmbCoef(2:64,:);

EC = sum(S(:)==Th)+sum(S(:)==-Th);%计算振幅为1的系数个数

end