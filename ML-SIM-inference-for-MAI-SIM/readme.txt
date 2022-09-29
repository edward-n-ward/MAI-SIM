This folder contains the code for ML-SIM
reconstructions of SIM data

The main file is eval.py which will 
run batched reconstructions of multi-
page .tif files. This method is a 
fully convolutional network and as such
will accept any size of input image. There
is also a lighter model available and
ideally suited to real-time reconstructions.
Example scipts are also available to show
the effects of normalisation methods on
the input data. 

A new method is also available based on a 
video transformer network. Details for 
opperation and source code can be found
in the repository for this method:
https://github.com/charlesnchr/VSR-SIM
