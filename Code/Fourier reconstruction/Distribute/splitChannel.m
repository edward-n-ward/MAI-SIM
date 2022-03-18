a_num =3;
p_num = 3;
[files, path] = uigetfile('D:\SIM Data\17-03-2022\to process\*.tif','multiselect','on');
for imFile = 1:length(files) 
   disp('Starting Stack')
   
try
    
inFile = fullfile(path,files{imFile});
r_top = 140; r_left = 1464;
g_top = 140; g_left = 784;
b_top = 140; b_left = 106;

x_width = 511;
y_width = 511;

outFileR = strcat(inFile(1:(end-28)),'_Split.tif');
r = 1;
g = 1;
b = 1;

info = imfinfo(inFile);
n = floor(length(info)/(a_num*p_num));

% Load and save red images
msg = strcat('Images');
fTIF = Fast_Tiff_Write(outFileR,0.125,0);
PSF_edge = fspecial('gaussian',5,40);
ind = 1;
for i = 1:n 
    textwaitbar(i, n, msg); 
    % Load and process green images
    if g == 1
        for a = 0:(a_num-1)
        for p = 0:(p_num-1)
            loaded = imread(inFile,ind+((a*a_num)+p))';
            green = loaded(g_left:g_left+x_width,g_top:g_top+y_width);
%             if b ==1
%             blue = loaded(b_left:b_left+x_width,b_top:b_top+y_width);
%             green = green-0.02*blue;
%             green(green<0)=0;
%             else
                 green = green-min(green(:));
%             end
            
            temp = edgetaper(green,PSF_edge);
            fTIF.WriteIMG(green);
        end
        end
    end
    % Load and process red images
    if r == 1
        for a = 0:(a_num-1)
        for p = 0:(p_num-1)
            loaded = imread(inFile,ind+((a*a_num)+p))';
            red = loaded(r_left:r_left+x_width,r_top:r_top+y_width);
            
%             if g ==1
%             green = loaded(g_left:g_left+x_width,g_top:g_top+y_width);
%             red = red-0.02*green;
%             red(red<0)=0;
%             else
                red = red-min(red(:));
%             end
            
            red = edgetaper(red,PSF_edge);
            fTIF.WriteIMG(red);
        end
        end
    end
    % Load and process blue images
    if b == 1
        for a = 0:(a_num-1)
        for p = 0:(p_num-1)
            loaded = imread(inFile,ind+((a*a_num)+p))';
            blue = loaded(1404:1404+x_width,94:94+y_width);
            blue = blue-min(blue(:));
                        
            blue = edgetaper(blue,PSF_edge);
            fTIF.WriteIMG(blue);
        end
        end
    end    
    ind = ind+(a_num*p_num);
end
fTIF.close;
catch
    disp('Failed to save stack');
try
    fTIF.close;
catch
    disp('Failed to close tiff reader');
end
end
end
        
%% Waitbar functions

function textwaitbar(i, n, msg)
% A command line version of waitbar.
% Usage:
%   textwaitbar(i, n, msg)
% Input:
%   i   :   i-th iteration.
%   n   :   total iterations.
%   msg :   text message to print.
%
% Date      : 05/23/2019
% Author    : Xiaoxuan He   <hexxx937@umn.edu>
% Institute : University of Minnesota
%
% Previous percentage number.
persistent i_prev_prct;
% Current percentage number.
i_prct = floor(i ./ n * 100);
% Print message when counting starts.
if isempty(i_prev_prct) || i_prct < i_prev_prct
    i_prev_prct = 0;
    S_prev = getPrctStr(i_prev_prct);
    
    fprintf('%s: %s',msg, S_prev);
end
% Print updated percentage.
if i_prct ~= i_prev_prct
    S_prev = getPrctStr(i_prev_prct);
    fprintf(getBackspaceStr(numel(S_prev)));
    
    S = getPrctStr(i_prct);
    fprintf('%s', S);
    
    i_prev_prct = i_prct;
end
% Clear percentage variable.
if i_prct == 100
    fprintf(' Done.\n');
    clear i_prev_prct;
end
end
function S = getPrctStr(prct)
S = sprintf('%d%%  %s',prct,getDotStr(prct));
if prct < 10
    S = ['  ',S];
elseif prct < 100
    S = [' ',S];
end
end
function S = getDotStr(prct)
S = repmat(' ',1,10);
S(1:floor(prct/10)) = '.';
S = ['[',S,']'];
end
function S = getBackspaceStr(N)
S = repmat('\b',1,N);
end

    
    