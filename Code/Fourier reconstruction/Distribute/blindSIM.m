% Blind-SIM reconstruction
clear all
close all

%% inputs
K = 60;
path = ''; % Path to image stack
OTF_path = ''; % Path to OTF
mode = 1; % Use 1 for division, use 2 for inverse matrix 

a_num = 3; % Number of angles
p_num = 3; % Number of phases 

% Reconstruction parameters

x = 1;

beta = 15; 
alpha = 0.095; 

theta = 2; 
t=0.1; 
sigma = 3e4;

% Load images
data = double(getNormTifStack(path,PSF));
data = data./max(data(:));
data_f = ft(data);
OTF = double(imread(OTF_path)); 
OTF = OTF./max(OTF(:));
PSF = ft(OTF);
PSF = PSF./max(PSF(:));
PSF=PSF.^0.9;
OTF = ft(PSF);
OTF = OTF./max(OTF(:));

% get patterns
pattern = getPattern(3,3,path,561,86,1.2,4,0.4,1);




w = zeros(size(data));
page = 1;
for a = 1:a_num
    for p = 1:p_num
        w(:,:,page) = pattern(:,:,a,p);
        page = page+1;
    end
end

w = w+0.3;
w = w./max(w(:));


Im = mean(data,3);
p = zeros(size(Im));
g = conj(OTF).*OTF;
b=(2*g+2*(beta/alpha));

q = zeros(size(data));
w_f = zeros(size(w));
q_f = zeros(size(w));
M = size(data,3);
for i =1:M
    q(:,:,i) = Im.*w(:,:,i);
    q_f(:,:,i) = ft(q(:,:,i));
    w_f(:,:,i) = ft(w(:,:,i));
end
I = mean(pattern(:));


rho = zeros(size(Im));
for i = 1:x
for m = 1:M
    for k = 1:K
        d_k = (w_f(:,:,M)-2*(OTF.*data_f(:,:,m)-(g+beta).*(q_f(:,:,m))))./b;
        q_f(:,:,m) = q_f(:,:,m)-theta*t*d_k;

        a_k = ift(w_f(:,:,m)+sigma*(q_f(:,:,m)-2*t*d_k));
        w(:,:,m) = (1-theta)*w(:,:,m)+theta*min(a_k,alpha);
        w_f(:,:,m) = ft(w(:,:,m));
    end

rho = rho+1/(M*x).*ift(q_f(:,:,m))./I;


end
rho = rho./max(rho(:));
end

svpath = strcat(path(1:end-4),'_blindSIM.tif');
imwrite(uint16(65000*(rho./max(rho(:)))),svpath);
disp('DONE!');


%% Helper functions
function A = ft(A)
    A = fftshift(fft2(fftshift(A)));
end

function A = ift(A)
    A = real(ifftshift(ifft2(ifftshift(A))));
    A(A<0) = 0;
end

function TifStack = getNormTifStack(path,PSF)

    % Image loading constants
    PSF_edge = fspecial('gaussian',5,40);

    % Image reading loop
    TifStack = double(imread(path, 1));  
    for i = 1:9
        load = double(imread(path,i));
        [X, Y] = size(load);
        lim = min(X,Y);
        load=load(1:lim,1:lim);
        load = load./max(load(:));
        load=edgetaper(load,PSF_edge);
        load = deconvlucy(load,PSF,4);
        TifStack(:,:,i) = load;
        
    end
    
end