% prototype program for SIM reconstruction using inverse matrix based phase estimaton algorithm

close all;
clear all

%% read image file
a_num=3;% number of pattern orientations
p_num=3;% phase shift times for each pattern orientation
filepath = 'D:\SIM Data\04-03-2022\tubGreen_2\fixed good phases.tif';

%% parameter of the detection system
lambda=590;% fluorescence emission wavelength (emission maximum). unit: nm
psize=85; 
NA=1.2;

%% parameter for reconstruction
iter = 5;
mask_factor=0.4;

%% Load images
edges = linspace(0,2^16,(2^8));
for ii=1:a_num
    for jj=1:p_num
        load = double(imread(filepath,((ii-1)*3+jj)));
        [X, Y] = size(load);
        lim = min(X,Y);
        load=load(1:lim,1:lim);
        [counts,edges] = histcounts(load,edges);
        background = edges(counts==max(counts));
        load = load-background; load(load<0)=0;
        load = double(65536.*(load./max(load(:))));
        noiseimage(:,:,ii,jj)=load;
    end
end

[xsize,ysize]=size(noiseimage(:,:,1,1));
[Y,X]=meshgrid(1:ysize,1:xsize);

PSF_edge = fspecial('gaussian',5,40);
for ii=1:a_num
    for jj=1:p_num
        noiseimage(:,:,ii,jj)=edgetaper(noiseimage(:,:,ii,jj),PSF_edge);
    end
end


xc=floor(xsize/2+1);% the x-coordinate of the center
yc=floor(ysize/2+1);% the y-coordinate of the center
yr=Y-yc;
xr=X-xc;
R=sqrt((xr).^2+(yr).^2);% distance between the point (x,y) and center (xc,yc)

%% Generate the PSF
pixelnum=xsize;
rpixel=NA*pixelnum*psize/lambda;
cutoff=round(2*rpixel);% cutoff frequency
ctfde=ones(pixelnum,pixelnum).*(R<=rpixel);
ctfdeSignificantPix=numel(find(abs(ctfde)>eps(class(ctfde))));
ifftscalede=numel(ctfde)/ctfdeSignificantPix;
apsfde=fftshift(ifft2(ifftshift(ctfde)));
ipsfde=ifftscalede*abs(apsfde).^2;
OTFde=real(fftshift(fft2(ifftshift(ipsfde))));
clear apsfde ctfde temp X Y

%% filter/deconvolution before using noiseimage

for ii=1:a_num
    for jj=1:p_num

       noiseimage(:,:,ii,jj)= deconvlucy(noiseimage(:,:,ii,jj),ipsfde,iter);
       noiseimage(:,:,ii,jj)=noiseimage(:,:,ii,jj).*(noiseimage(:,:,ii,jj)>0);
    end
end
widefield=sum(sum(noiseimage,4),3);
separated_FT=zeros(xsize,ysize,a_num,3);
noiseimagef=zeros(size(noiseimage));
for ii=1:a_num
    re0_temp=zeros(xsize,ysize);
    rep_temp=zeros(xsize,ysize);
    rem_temp=zeros(xsize,ysize);
    modulation_matrix=[1,1/2*exp(-1i*(pi*0)),1/2*exp(1i*(pi*0));...
                       1,1/2*exp(-1i*(pi*2/3)),1/2*exp(1i*(pi*2/3));...
                       1,1/2*exp(-1i*(pi*4/3)),1/2*exp(1i*(pi*4/3))];
    matrix_inv=inv(modulation_matrix);

    for jj=1:p_num
        noiseimagef(:,:,ii,jj)=fftshift(fft2(noiseimage(:,:,ii,jj)));
        re0_temp=matrix_inv(1,jj)*noiseimagef(:,:,ii,jj)+re0_temp;
        rep_temp=matrix_inv(2,jj)*noiseimagef(:,:,ii,jj)+rep_temp;
        rem_temp=matrix_inv(3,jj)*noiseimagef(:,:,ii,jj)+rem_temp;
    end

    separated_FT(:,:,ii,1)=re0_temp;
    separated_FT(:,:,ii,2)=rep_temp;
    separated_FT(:,:,ii,3)=rem_temp;
end
clear re0_temp rep_temp rem_temp noiseimage

fmask=double(sqrt(xr.^2+yr.^2)>cutoff*mask_factor);
show_initial_result_flag = 0;
[shiftvalue,~]=frequency_est_tirf_v2(separated_FT,0.008,fmask,show_initial_result_flag,mask_factor*cutoff);
clear separated_FT


for ii=1:a_num
    shiftvalue(ii,2,:)=shiftvalue(ii,2,:)-shiftvalue(ii,1,:);
    shiftvalue(ii,3,:)=shiftvalue(ii,3,:)-shiftvalue(ii,1,:);
    shiftvalue(ii,1,1)=0;
    shiftvalue(ii,1,2)=0;
end

%% phase correction with inverse matrix based algorithm
search_range=0.6; %the max radius in the local search algorithm

%obtain a more precise estimation of the period and the directon of sinusodial pattern
[ precise_shift,~] = precise_frequency_tirf(noiseimagef,shiftvalue,search_range);

[inv_phase] = separation_matrix_correction_v3(noiseimagef,precise_shift,OTFde);

%% auto-correlation based algorithm
auto_phase=zeros(a_num,p_num);

for ii=1:a_num
    for jj=1:p_num
        f_temp=exact_shift(noiseimagef(:,:,ii,jj),...
        [-precise_shift(ii,2,1),-precise_shift(ii,2,2)],1);
     
        auto_phase(ii,jj)=angle(sum(sum(conj(noiseimagef(:,:,ii,jj)).*f_temp)));
    end
end

% my_phase_temp=mod(-inv_phase,2*pi);
% my_phase_auto=mod(-auto_phase,2*pi);
% my_phase_cc=mod(cc_phase,2*pi);

inv_phase=auto_phase;
patterns = zeros(size(noiseimagef));
xn=xr+xsize/2; xn = 2*pi*xn./max(xn(:));
yn=yr+ysize/2; yn = 2*pi*yn./max(yn(:));
for a = 1:a_num
    for p =1:p_num
        phase = inv_phase(((a-1)*a_num)+p);
        patterns(:,:,a,p)=sin(xn*precise_shift(a,2,1)+yn*precise_shift(a,2,2)+phase);
        imwrite(uint16(65000*patterns(:,:,a,p)),'patterns.tif','writemode','append');
    end
end
