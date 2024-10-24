clear all;clc;
warning off;
addpath('npy-matlab-master/npy-matlab')
addpath('MMWeng');
% E:\codes\database\ucid\cnnjpeg

% jpgdir = 'E:/codes/database/ucid/100/';
% npydir = 'E:/codes/CNN_RDH_JPEG/cnnjpeg/image/100/ucid/';

% dirList = dir(jpgdir);
% fileCount = length(dirList) - 2;
fileCount = 100;
% fileCount = 10;

payloads = [3000,6000,9000,12000,15000];
QFs = [70,80,90];
PSNR_all = cell(5,3);
INC_all = cell(5,3);

for jj = 1:3
    QF = QFs(jj);
    for i = 0 : fileCount - 1
        jpgpath = strcat( 'E:/codes/database/BOSSBase/BOSSbase100_1/',num2str(QF),'/',num2str(i),'.jpg');
        jpgObj = jpeg_read(jpgpath);
        jpgCoef = jpgObj.coef_arrays{1};
        
        
        load(strcat('default_gray_jpeg_obj_', num2str(QF),'.mat'));
        stegoObj = default_gray_jpeg_obj;
        stegoObj.coef_arrays{1} = jpgCoef;
        stegoObj.optimize_coding = 1;
        stegoJPEG = strcat( 'E:/codes/database/BOSSBase/BOSSbase100/',num2str(QF),'/',num2str(i),'.jpg');
        jpeg_write(stegoObj,stegoJPEG);
    end
end


