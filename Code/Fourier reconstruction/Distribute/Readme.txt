
Readme for Frequency-space based SIM reconstruction
*************************************************************************

The folder contains the code necessary to reconstruct Athesim
images using the inverse matrix approach created by Cao, R et al. [1]
For a detailed description of the code please refer to the original publication.

This folder includes three ways to run the reconstruction code:
1) The function SIM_main_v2.m is the originally reported function and should be considered
the most stable and reliable

2) The function Batch_SIM_main_v2.m includes a handy wrapper around SIM_main_v2.m 
to enable batch reconstructions of multiple multi-page .tif images

3) The MATLAB app SIMReconstruction.mlapp offers a GUI for running the reconstruction
and includes code to allow for more advanced SIM data analysis and pre-/post-processing.
While this app may be useful it is undergoing continual update and bug fixes and should
not be consider reliable for all reconstructions/ inputs.

Also included are several helper functions, which can be used to rapidly write large .tif
files, split multi-colour images acquired with an optosplit and unroll a stack of SIM 
images acquired with the 12-frame rolling method.


References:

[1] Ruizhi Cao, Youhua Chen, Wenjie Liu, Dazhao Zhu, Cuifang Kuang, Yingke Xu, and Xu Liu.
"Inverse matrix based phase estimation algorithm for structured illumination microscopy," Biomed. Opt. Express 9, 5037-5051 (2018)
