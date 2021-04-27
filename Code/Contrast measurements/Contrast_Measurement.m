% Extract the SIM pattern contrast from a stack of bead images
% Laurie Young 2016
% Modified by Romain Laine Sept. 2016
% Modified again by Marcus Fantham April 2017

close all
clear all
clc

DefaultPath = 'C:\Data\sim\';
[Filename, Pathname, Filterspec] = uigetfile({'*.cxd'; '*.tif'},'Choose a file...', DefaultPath);
FullFileName = [Pathname Filename];

%%
t_start = tic;
% profile on

% % File            = 'C:\Users\rfl30\DATA raw\SIM data\2016_08_31_tirf-sim_setup\p12_488\p12_120_1';
% % Extension       = '.cxd';
% % File_Name = [File Extension];

Num_Phases      = 12;
minV            = -1;
maxV            = 1;

Threshold       = 1/10; % only used when using FastPeakFind
MaxArea         = 49; % in pixels, for getting rid of big objects (multiple beads or crap)

Correction_factor = 0.629; % empirically found that actual and measured contrast are related by a linear relationship with AC = 0.629*MC


% ------------------- Methods -----------------------
% Segmentation_method = 'FastPeakFind';
Segmentation_method = 'OtsuThresholding';

% Contrast_calc_method = 'MinMax';
% Contrast_calc_method = 'NonLinearFitting';  % painfully slow...
Contrast_calc_method = 'LinearRegression';

if Filterspec == 1
    reader = bfGetReader(FullFileName);
    omeMeta = reader.getMetadataStore();
    Width = omeMeta.getPixelsSizeX(0).getValue();
    Height = omeMeta.getPixelsSizeY(0).getValue();
    nFrames = reader.getImageCount();
    n_cycles = 2; % the 9 pixels phase shifts correspond to 2 cycles
elseif Filterspec == 2
    info = imfinfo(FullFileName);
    Width = info(1).Width;
    Height = info(1).Height;
    nFrames = length(info);
    n_cycles = 2; % the simulation only does one cycle
end

% loop through stack
disp('--------------------------------');
n_Steps = nFrames/Num_Phases;
disp(['Number of steps: ', int2str(n_Steps)]);
disp('--------------------------------');

Mean_Bead_Contrast = zeros(n_Steps,1);
STD_Bead_Contrast = zeros(n_Steps,1);


h_wait = waitbar(0, 'Please wait...');
for n = 1:n_Steps
    waitbar(n/n_Steps, h_wait);
    %     disp(['n = ' int2str(n)]);
    
    % Sum the correct Num_Phases images and find the bead positions
    Imgs = zeros(Num_Phases, Height, Width);
    
    if Filterspec == 1
        for i = 1:Num_Phases
            Imgs(i,:,:) = double(bfGetPlane(reader, (n-1)*Num_Phases + i));
        end
    elseif Filterspec == 2
        for i = 1:Num_Phases
            Imgs(i,:,:) = imread(FullFileName, (n-1)*Num_Phases + i);
        end
    end
    % Calculate the average image
    Img_Av = squeeze(sum(Imgs,1)/Num_Phases);
    
    % --------- Using FastPeakFind ----------
    if strcmp(Segmentation_method, 'FastPeakFind')
        % Find peaks in the summed image using weighted centroids
        pos = FastPeakFind(Img_Av.*(Img_Av>max(Img_Av(:))*Threshold));
        n_beads = length(pos)/2;
        
        % Loop through phase images and get intensity variation for each bead
        Amp = zeros(n_beads, Num_Phases);
        for i = 1:Num_Phases
            ThisImg = squeeze(Imgs(i,:,:));
            Amp(:,i) = squeeze(ThisImg(sub2ind(size(ThisImg),pos(2:2:end),pos(1:2:end))));
        end
        
        % --------- Using Otsu Thresholding -----------
    elseif strcmp(Segmentation_method, 'OtsuThresholding')
        
        % Threshold and binarize
        BW = imbinarize(Img_Av/max(Img_Av(:)),'global');
        BG = mean(Img_Av(~BW)); % Compute the background value
        
        % Clear the ones on the border
        BW = imclearborder(BW);
        
        % Get rid of the large ones
        cc = bwconncomp(BW);
        stats = regionprops(cc, 'Area');
        idx = find([stats.Area] < MaxArea);
        BW = ismember(labelmatrix(cc), idx);
        cc = bwconncomp(BW);
        n_beads = cc.NumObjects;
        
        % Loop through phase images and get intensity variation for each bead
        Amp = zeros(n_beads, Num_Phases);
        for i = 1:Num_Phases
            ThisImg = squeeze(Imgs(i,:,:));
            SpotProps = regionprops(cc, squeeze(Imgs(i,:,:)-BG), 'MeanIntensity');
            Amp(:,i) = [SpotProps.MeanIntensity];
        end
    end
    
    % Display the number of beads found
    disp([int2str(n) ': number of beads in image = ' int2str(n_beads)]);
    
    % Extract contrast -------------------------------
    Contrasts = zeros(n_beads,1);
    
    if strcmp(Contrast_calc_method, 'MinMax') % ---------------------------
        for k = 1:n_beads
            Contrasts(k) = (max(Amp(k,:)) - min(Amp(k,:)))/(max(Amp(k,:)) + min(Amp(k,:)));
        end
    elseif strcmp(Contrast_calc_method, 'NonLinearFitting') % -------------
        Phi = n_cycles*2*pi*(1:Num_Phases)/Num_Phases; % we go over 2 cycles so Phi goes from 0 to 4*pi !
        for k = 1:n_beads
            % Fit sinusoid to intensity variation at each bead position
            [xData, yData] = prepareCurveData( Phi, Amp(k,:));
            
            D_init = mean(Amp(k,:));
            A_init = max(Amp(k,:)) - D_init;
            B_init = 1;
            C_init = 0;
            
            ft = fittype( 'A*sin(B*x-C)+D', 'independent', 'x', 'dependent', 'y' );
            opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
            opts.Display = 'Off';
            opts.StartPoint = [A_init B_init C_init D_init];
            opts.Lower = [0 0 -Inf 0];
            opts.Upper = [Inf 2 Inf Inf];
            
            % Perform fit
            [fitresult, gof] = fit( xData, yData, ft, opts );
            
            % Correct way to calculate the contrast of a sine wave
            Contrasts(k) = fitresult.A/fitresult.D;
            %             Bead_Freq(n,k) = fitresult.B;
            %             Sine_fit(n,k,:) = fitresult.A.*sin(fitresult.B*phases - fitresult.C)-fitresult.D;
            %             Bead_Contrast_fit(n,k) = (max(Sine_fit(n,k,:)) - min(Sine_fit(n,k,:)))  ./ (max(Sine_fit(n,k,:)) + min(Sine_fit(n,k,:)));
        end
    elseif strcmp(Contrast_calc_method, 'LinearRegression') % -------------
        for k = 1:n_beads
            
            % Prepare variables for linear regression
            Y = Amp(k,:)';
            Phi = n_cycles*2*pi*(1:Num_Phases)/Num_Phases; % we go over 2 cycles so Phi goes from 0 to 4*pi !
            X = [ones(size(Phi'))  cos(Phi)'  sin(Phi)'];
            % Perform linear regression using \ operator
            a = X\Y;
            % Calculate contrast from the results
            B_lin = a(1);
            A_lin = sqrt(a(2)^2+a(3)^2);
            
            Contrasts(k) = sqrt(a(2)^2+a(3)^2)/a(1);
            
        end
    end
    
    Mean_Bead_Contrast(n) = mean(Contrasts/Correction_factor);
    STD_Bead_Contrast(n) = std(Contrasts/Correction_factor);
    
end
close(h_wait);

%%
voltages = minV + (0:n_Steps-1)*(maxV-minV)/(n_Steps-1);
h_results = figure('Color','white');
% Plot located bead centres
% subplot(1,3,1);
% imagesc(Img_Av);
% axis image
% colormap(gray);
% hold on
% plot(pos(1:2:end),pos(2:2:end),'r.');
% hold off

% subplot(1,3,2);
% if do_fit
%     plot(voltages, mean(Bead_Contrast_fit,2),'LineWidth',2);
% end

% hold on
plot(voltages,Mean_Bead_Contrast,'LineWidth',2);
hold on
errorbar(voltages, Mean_Bead_Contrast,STD_Bead_Contrast,'k.');
hold off
axis square
box on
% xlabel('LCVR Voltage (V)');
xlabel('Voltage V');
ylabel('Contrast');
xlim([minV maxV]);
ax2=gca;
ax2.FontSize = 12;
ax2.XLabel.FontSize = 12;
ax2.YLabel.FontSize = 12;
grid on


%%
[path, name, ~] = fileparts(FullFileName);
saveas(h_results, [path '\' name '.png']);
% set(gcf,'Units','centimeters');
% savefig([path '\' name '.fig']);

if Filterspec == 1
    reader.close();
    clear omeMeta reader
end
% save([File '.mat']);


disp('--------------------------------');
toc(t_start);
% profile viewer
