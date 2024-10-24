clear all;clc;
warning off;
addpath('npy-matlab-master/npy-matlab')

QF = 70;
fileCount = 10;
jpgdir = 'E:/codes/CNN_RDH_JPEG/cnnjpeg/standimage/';
jpglist = dir(strcat(jpgdir,num2str(QF)));
for i = 3 : 12
    jpgpath = strcat( 'E:/codes/CNN_RDH_JPEG/cnnjpeg/standimage/',num2str(QF),'/',jpglist(i).name);
    jpgObj = jpeg_read(jpgpath);
    jpgCoef = jpgObj.coef_arrays{1};
    s = jpgCoef;
    aep = jpgCoef;
    for n = 1:512
        for m = 1:512
            if abs(aep(n,m)) == 1
                aep(n,m) = 1;
            else
                aep(n,m) = 0;
            end
        end
    end
    savepath1 = strcat('E:/codes/CNN_RDH_JPEG/cnnjpeg/standimage/70_originalcoef/',jpglist(i).name, '.npy');
    writeNPY(aep, savepath1);
    
    prop = zeros(8,8);
%     for n = 1:512
%         for m = 1:512
%             if mod(n,8) == 1 && mod(m,8) == 1
%                 jpgCoef(n,m) = 0;
%             end
%         end
%     end
    for n = 1:512
        for m = 1:512
            if jpgCoef(n,m) == 0
                prop(mod(n-1,8)+1,mod(m-1,8)+1) = prop(mod(n-1,8)+1,mod(m-1,8)+1) + 1;
            end
        end
    end
    for n = 1:8
        for m = 1:8
            prop(n,m) = min(prop(n,m), 4095);
            prop(n,m) = max(prop(n,m), 1);
        end
    end
    lambda  = -2*log(1-prop/4096);
    pz = CDF(lambda,0.5)-CDF(lambda,-0.5);
    pc = 2*(CDF(lambda,-0.5)-CDF(lambda,-1.5));
    ppp = pc ./ ( 1 - pz);
    for n = 1:512
        for m = 1:512
            if mod(n,8) == 1 && mod(m,8) == 1
                jpgCoef(n,m) = 0;
            else
                if jpgCoef(n,m) ~= 0
                    jpgCoef(n,m) = ppp(mod(n-1,8)+1,mod(m-1,8)+1);
                end
            end
        end
    end
    savepath = strcat('E:/codes/CNN_RDH_JPEG/cnnjpeg/standimage/70_predictioncoef/',jpglist(i).name, '.npy');
    writeNPY(jpgCoef, savepath);
    



end