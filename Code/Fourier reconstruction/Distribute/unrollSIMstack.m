
[files, path] = uigetfile('D:\SIM Data\25-02-2022\to process\*.tif','multiselect','on');
for imFile = 1:length(files) 

inFile = fullfile(path,files{imFile});


anum = 3;
pnum = 3;

outFile = strcat(inFile(1:(end-4)),'_Unrolled.tif');
info = imfinfo(inFile);
n = length(info);
frames = floor(n/6)+1 - pnum;

for i = 1: anum*pnum
   load = imread(inFile,i);
   imwrite(load,outFile,'writemode','append');     
end

msg = strcat('Saving image: ',files{imFile});
fTIF = Fast_Tiff_Write(outFile,0.125,0);

imNum = i-pnum;
try
for f = 2:frames
for a = 1:anum
for p = 1:pnum

    textwaitbar(f, frames, msg);
    imNum=imNum+1;
    load = imread(inFile,imNum);
    fTIF.WriteIMG(load');
    
end
end
    imNum = imNum-pnum;
end
catch
    
    disp('Finished Processing Stack')
end
fTIF.close;    
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

    
    