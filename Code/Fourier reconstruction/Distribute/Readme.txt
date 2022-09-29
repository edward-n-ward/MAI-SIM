
Readme for Frequency-space based SIM reconstruction
*************************************************************************

The folder contains the code necessary to reconstruct Athesim
images using the inverse matrix approach created by Cao, R et al. [1]
For a detailed description of the code please refer to the original publication.

This folder includes three ways to run the reconstruction code:
1) The function SIM_main_v2.m is a lightly adapted version of the originally reported 
function and is likely the most stable and reliable. Extra functionality has been
added for more flexible pre- and post-processing. The code will automatically save
two .tif files, the SIM reconstruction and the widefield image and display the 
resulting reconstructions. Typical execution time for the code is a few seconds per 
image.

2) The function Batch_SIM_main_v2.m includes a handy wrapper around SIM_main_v2.m 
to enable batch reconstructions of multiple multi-page .tif images although pre-
and post- processing are not available.

3) The script blindSIM.m is used to perfrom the blind SIM reconstructions.[2] 

Also included are several helper functions, which can be used to rapidly write large 
.tif files, split multi-colour images acquired with an optosplit and unroll a stack of 
SIM images acquired with the 12-frame rolling method.


References:

[1] Ruizhi Cao, Youhua Chen, Wenjie Liu, Dazhao Zhu, Cuifang Kuang, Yingke Xu, and Xu Liu.
"Inverse matrix based phase estimation algorithm for structured illumination microscopy" 
Biomed. Opt. Express 9, 5037-5051 (2018)
[2] Labouesse, S. et al. Joint Reconstruction Strategy for Structured Illumination 
Microscopy With Unknown Illuminations. IEEE Trans. Image Process. 26, 2480–2493 (2017).
