function [widefield, reconstructed, FT] = reconWicker(noiseimage, lambda, psize, NA, ...
    wiener_factor, mask_factor,preFilter,postFilter,highNoise,n_iter,apo,apoWidth,upsample, ... 
    stripe, stripeWidth,shift,shiftValues,debug)

a_num=3;% number of pattern orientations
p_num=3;% phase shift times for each pattern orientation


%% visualization option
show_initial_result_flag=0;
% the phase-only correlation result without correction will
% be displayed when show_initial_result_flag equals 1

show_corrected_result_flag=0;
% the phase-only correlation result after correction will
% be displayed when show_corrected_result_flag equals 1


[xsize,ysize]=size(noiseimage(:,:,1,1));
[Y,X]=meshgrid(1:ysize,1:xsize);

PSF_edge = fspecial('gaussian',6,15);
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
rpixel=NA*pixelnum*psize/(lambda);
cutoff=round(2*rpixel);% cutoff frequency
ctfde=ones(xsize,ysize).*(R<=rpixel);
ctfdeSignificantPix=numel(find(abs(ctfde)>eps(class(ctfde))));
ifftscalede=numel(ctfde)/ctfdeSignificantPix;
apsfde=fftshift(ifft2(ifftshift(ctfde)));
ipsfde=ifftscalede*abs(apsfde).^2;
OTFde=real(fftshift(fft2(ifftshift(ipsfde))));
clear apsfde ctfde temp X Y
%% filter/deconvolution before using noiseimage
%pre-deconvolution. It can be applied to suppress noises in experiments

widefield=sum(sum(noiseimage,4),3);
%Widefield is the sum of all of the images
if preFilter == 1
    if highNoise == 1
        widefield=deconvlucy(widefield,ipsfde,n_iter);
    else
        widefield=quasi_wnr(OTFde,widefield,wiener_factor^2);
    end
end

widefield=widefield.*(widefield>0);

for ii=1:a_num
    for jj=1:p_num
        if preFilter == 1
            if highNoise == 1
                noiseimage(:,:,ii,jj)=deconvlucy(noiseimage(:,:,ii,jj),ipsfde,3);                
            else
                noiseimage(:,:,ii,jj)=quasi_wnr(OTFde,squeeze(noiseimage(:,:,ii,jj)),wiener_factor^2);                
            end
        end
        

        noiseimage(:,:,ii,jj)=noiseimage(:,:,ii,jj).*(noiseimage(:,:,ii,jj)>0);
    end
end

%% Define the OTF apodisation
%OTFfilt is either blank or a negative exponential

if apo == 1
    OTFfilt = 1 - exp(-apoWidth*(R.^2));
else
    OTFfilt = ones(size(R));
end

widefield=widefield./max(widefield(:))*max(noiseimage(:));

%% Initialise values
%OTFfilt is either blank or a negative exponential

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
clear re0_temp rep_temp rem_temp 

%% First estimate of frequency components
fmask=double(sqrt(xr.^2+yr.^2)>cutoff*mask_factor);

if shift == 0

[shiftvalue,~]=frequency_est_tirf_v2(separated_FT,0.008,fmask,show_initial_result_flag,mask_factor*cutoff);
clear separated_FT


for ii=1:a_num
    shiftvalue(ii,2,:)=shiftvalue(ii,2,:)-shiftvalue(ii,1,:);
    shiftvalue(ii,3,:)=shiftvalue(ii,3,:)-shiftvalue(ii,1,:);
    shiftvalue(ii,1,1)=0;
    shiftvalue(ii,1,2)=0;
end

else
    shiftvalue = zeros(3,3,2);
    shiftvalue(:,2,1) = shiftValues(:,1);
    shiftvalue(:,2,2) = shiftValues(:,2);
    
    shiftvalue(:,3,1) = -shiftValues(:,1);
    shiftvalue(:,3,2) = -shiftValues(:,2);
    
end

%% Second estimate of frequency components

search_range=0.4;%the max radius in the local search algorithm
[ precise_shift,~] = precise_frequency_tirf(noiseimagef,shiftvalue,search_range);


%% phase correction with inverse matrix based algorithm

% estimation the phase of each pattern
% [my_phase] = separation_matrix_correction_v2(noiseimagef,precise_shift,OTFde);

[inv_phase] = separation_matrix_correction_v3(noiseimagef,precise_shift,OTFde);


%% auto-correlation based algorithm
% estimation the phase of each pattern

auto_phase=zeros(a_num,p_num);

for ii=1:a_num
    for jj=1:p_num
        f_temp=exact_shift(noiseimagef(:,:,ii,jj),...
        [-precise_shift(ii,2,1),-precise_shift(ii,2,2)],1);
     
        auto_phase(ii,jj)=angle(sum(sum(conj(noiseimagef(:,:,ii,jj)).*f_temp)));
    end
end

auto_phase=mod(-auto_phase,2*pi);

% inv_phase=auto_phase;
% reconstruct with phases determined by the auto-correlation method


%% separate different frequency component

n_filt = 1 - exp(-stripeWidth*(R.^2));

separated_FT=zeros(xsize,ysize,a_num,3);% store different bands of frequency component
for ii=1:a_num
    re0_temp=zeros(xsize,ysize);
    rep_temp=zeros(xsize,ysize);
    rem_temp=zeros(xsize,ysize);
    mi=0.5;
    modulation_matrix=[1,mi*exp(-1i*(inv_phase(ii,1))),mi*exp(1i*(inv_phase(ii,1)));...
                       1,mi*exp(-1i*(inv_phase(ii,2))),mi*exp(1i*(inv_phase(ii,2)));...
                       1,mi*exp(-1i*(inv_phase(ii,3))),mi*exp(1i*(inv_phase(ii,3)))];

    matrix_inv=inv(modulation_matrix);
    for jj=1:p_num
        re0_temp=matrix_inv(1,jj)*noiseimagef(:,:,ii,jj)+re0_temp;
        rep_temp=matrix_inv(2,jj)*noiseimagef(:,:,ii,jj)+rep_temp;
        rem_temp=matrix_inv(3,jj)*noiseimagef(:,:,ii,jj)+rem_temp;
    end
     if stripe ==1
        separated_FT(:,:,ii,1)=re0_temp.*OTFfilt;
        separated_FT(:,:,ii,2)=rep_temp.*n_filt;
        separated_FT(:,:,ii,3)=rem_temp.*n_filt;
     else
        separated_FT(:,:,ii,1)=re0_temp.*OTFfilt;
        separated_FT(:,:,ii,2)=rep_temp;
        separated_FT(:,:,ii,3)=rem_temp;
     end
     
end

[~,noise_ratio]=frequency_est_tirf_v2(separated_FT,0.008,fmask,show_corrected_result_flag,mask_factor*cutoff);

clear noiseimagef
%% interpolate when necessary

if upsample == 1
    double_re=zeros(2*xsize,2*ysize,a_num,3);
    for ii=1:a_num
        for jj=1:3
            double_re((xsize-xc+2):(2*xsize-xc+1),(ysize-yc+2):(2*ysize-yc+1),ii,jj)=separated_FT(:,:,ii,jj);
        end
    end
    clear re_f
    separated_FT=double_re;
    clear double_re
    xsize=2*xsize;
    ysize=2*ysize;
    widefield=imresize(widefield,2,'bicubic');
    
    [Y,X]=meshgrid(1:ysize,1:xsize);
    xc=floor(xsize/2+1);
    yc=floor(ysize/2+1);
    yr=Y-yc;
    xr=X-xc;
    R=sqrt((xr).^2+(yr).^2);
    pixelnum=xsize;
    psize=psize/2;
    rpixel=NA*pixelnum*psize/lambda;

    ctfde=ones(pixelnum,pixelnum).*(R<=rpixel);
    ctfdeSignificantPix=numel(find(abs(ctfde)>eps(class(ctfde))));
    ifftscalede=numel(ctfde)/ctfdeSignificantPix;
    apsfde=fftshift(ifft2(ifftshift(ctfde)));
    ipsfde=ifftscalede*abs(apsfde).^2;
    OTFde=real(fftshift(fft2(ifftshift(ipsfde))));

end
clear re0_temp rem_temp rep_temp R X Y xr yr OTF_temp

OTF_n=zeros(size(OTFde));
OTF_nb=OTF_n;% OTF of reconstructed image
ft_true=zeros(size(separated_FT));
OTF_de_temp=abs(fftshift(fft2(ipsfde)));
OTFcirc=double(OTFde./(wiener_factor^2+OTFde));
for ii=1:a_num
    for jj=1:3
        if jj~=1
            ft_true(:,:,ii,jj)=exact_shift(separated_FT(:,:,ii,jj),...
                [precise_shift(ii,jj,1),precise_shift(ii,jj,2)],1);
            
        else
            ft_true(:,:,ii,jj)=separated_FT(:,:,ii,jj);
        end

        OTF_n=circshift(OTF_de_temp,[shiftvalue(ii,jj,1),shiftvalue(ii,jj,2)])+OTF_n;
        OTF_nb=circshift(OTFcirc,[shiftvalue(ii,jj,1),shiftvalue(ii,jj,2)])+OTF_nb;
    end
end
clear separated_FT
OTF_n=OTF_n./max(max(abs(OTF_n)));
OTF_nb=OTF_nb./max(max(OTF_nb));

psf_nb=abs(fftshift(fft2(OTF_nb))); %the efficient PSF for SIM with pre-deconvolution

psf_n=fftshift(ifft2(ifftshift(OTF_n)));
psf_n=abs(psf_n); %the efficient PSF for SIM without pre-deconvolution

mod_depth_temp=zeros(a_num,3);
reference=sum(ft_true(:,:,:,1),3);
for ii=1:a_num
    for jj=1:3
        if jj==1
            mask_temp=OTFcirc.*circshift(OTFcirc,[shiftvalue(ii,2,1),shiftvalue(ii,2,2)]);
            %with pre-deconvolution
     
            mod_depth_temp(ii,jj)=sum(sum(conj(reference).*ft_true(:,:,ii,jj).*mask_temp));
        else
            mod_depth_temp(ii,jj)=sum(sum(conj(reference).*ft_true(:,:,ii,jj)));
        end
        temp=mod_depth_temp(ii,jj)./abs(mod_depth_temp(ii,jj));
        if jj~=1
            ft_true(:,:,ii,jj)=ft_true(:,:,ii,jj)./temp;
        end
    end
end

%% estimate the modulation depth via cross-correlation

% % abs_mod=abs(mod_depth_temp);
% % illu_max=max(abs_mod,[],2);
% % for ii=1:a_num
% % abs_mod(ii,:)=abs_mod(ii,:)./illu_max(ii,1);
% % end
% % % the following is the best set of parameter among 10 sets. (Empirical value)
% % illu_max=illu_max./max(illu_max);
% % modulation_depth=mean(abs_mod(:,2:3),2);% modulation depth for each pattern orientation
% % noise_ratio=1./noise_ratio;
% % noise_suppress=noise_ratio(:,2)./max(noise_ratio(:,2));
% % 
% % for ii=1:a_num
% %     for jj=2:3
% %         myweight_factor=max([min([1/modulation_depth(ii),2.5]),1])...
% %         *noise_suppress(ii);
% %         ft_true(:,:,ii,jj)=ft_true(:,:,ii,jj)*myweight_factor;
% %     end
% % end

%% Reconstruction
FT_extended_per_angle=sum(ft_true,4);
FT_extended = sum(FT_extended_per_angle,3);
FT_extended(OTF_n<0.0001) = 0;
reconstructed_im=ifft2(ifftshift(FT_extended));
reconstructed=real(reconstructed_im).*(real(reconstructed_im>0));

if postFilter == 1
    if highNoise == 1
    reconstructed=deconvlucy(reconstructed,psf_nb,n_iter);
    else
    reconstructed=quasi_wnr(OTF_nb,reconstructed,wiener_factor^2);
    end
end
reconstructed = real(reconstructed);
reconstructed = 65535.*reconstructed./max(reconstructed(:));

FT = log(abs(FT_extended)+1); 
FT = 65535.*FT./max(FT(:));
widefield = 65535.*widefield./max(widefield(:));


end