import nidaqmx # microscope control
import numpy as np
import time
import torch.multiprocessing as mp
from pycromanager import Bridge
import torch
from models import *
import argparse

mp.freeze_support() 

## Grt ML-SIM params
def GetParams():
  opt = argparse.Namespace()

  # data
  opt.weights = 'C:/Users/SIM_ADMIN/Documents/GitHub/AtheSIM/ML-SIM-inference-for-AtheiSIM/DIV2K_randomised_3x3_20200317.pth' 
  
  # input/output layer options
  opt.task = 'simin_gtout'
  opt.scale = 1
  opt.nch_in = 9
  opt.nch_out = 1

  # architecture options 
  opt.model='rcan'#'model to use'  
  opt.narch = 0
  opt.n_resgroups = 3
  opt.n_resblocks = 10
  opt.n_feats = 96
  opt.reduction = 16
    
  return opt

## Convenience function
def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary

	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, vl in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = vl

	return new_state_dict

## Load ML-SIM model
def load_model():
    print('geting network params')
    opt = GetParams()
    print('building network')
    net = GetModel(opt)
    print('loading checkpoint')
    checkpoint = torch.load(opt.weights,map_location=torch.device('cuda'))
    if type(checkpoint) is dict:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    net.module.load_state_dict(state_dict)
    return net

## ML-SIM reconstruction
def ml_reconstruction(stack,output):
    net = load_model()
    while True:
        with torch.no_grad():
            if not stack.empty():
                pixels = stack.get()
                if isinstance(pixels, bool):
                        break
                else:
                    # run the reconstruction function 
                    data = torch.from_numpy(pixels)
                    data = torch.swapaxes(data,0,2)
                    data = data.unsqueeze(0)
                    data = data.type(torch.FloatTensor)
                    data = data.cuda()
                    data = data - torch.amin(data)
                    data = data/torch.amax(data)
                    sr = net(data.cuda())

                    sr = sr.cpu()
                    sr_frame = sr.numpy()
                    sr_frame = np.squeeze(sr_frame) 
                    output.put(sr_frame)
    
    output.put(False)

## Live view
def live_loop(stop_signal,output,exposure,opto,x1,y1,x2,y2):
    print('starting acquisition')
    stop_signal.put(True)
    with nidaqmx.Task() as VoltageTask, nidaqmx.Task() as CameraTask, Bridge() as bridge: # sorts out camera and microscope control
        voltages = np.array([0.95, 0.9508, 0.9516, 2.2, 2.2025, 2.205, 3.45, 3.454, 3.4548]) # microscope control values
        VoltageTask.ao_channels.add_ao_voltage_chan("Galvo_control/ao1")
        CameraTask.do_channels.add_do_chan("Galvo_control/port1/line3")
        core = bridge.get_core()

        if core.is_sequence_running():
            core.stop_sequence_acquisition() # stop the camera
            CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
            time.sleep(0.5/1000)
            CameraTask.write(False)

        core.start_continuous_sequence_acquisition(0) # start the camera

        CameraTask.write(True) # tell camera to take image
        time.sleep(exposure/1000)
        CameraTask.write(False)
        while core.get_remaining_image_count() == 0: #wait until picture is available
            time.sleep(0.001)
        result = core.get_last_tagged_image() # get image data into python        

        while True:
            status = stop_signal.get()
            if status == False:
                break
            else:
                stop_signal.put(True)
                if output.empty():
                    for i in range(9):
                        VoltageTask.write(voltages[i]) # move microscope
                        time.sleep(0.05)     

                        CameraTask.write(True) # start acquisition
                        time.sleep(exposure/1000)
                        CameraTask.write(False)
                        while core.get_remaining_image_count() == 0: # wait until image is available
                            time.sleep(0.001)                
                        result = core.get_last_tagged_image() # get image data into python
                        pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # reshape image data
                        pixels = pixels.astype('float64')
                        if opto == 1:
                            result = np.zeros((512,512,2))
                            result[:,:,0] = pixels[x1:x1+511,y1:y1+511]
                            result[:,:,1] = pixels[x2:x2+511,y2:y2+511]
                            output.put(result)
                        else:       
                            output.put(pixels)
                   
        core.stop_sequence_acquisition() # stop the camera
        CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
        time.sleep(0.5/1000)
        CameraTask.write(False)
        output.put(False)

## Live ML-SIM
def acquisition_loop(stop_signal,stack,exposure):
    print('starting acquisition')
    stop_signal.put(True)
    pixels = np.zeros((512,512,9))
    with nidaqmx.Task() as VoltageTask, nidaqmx.Task() as CameraTask, Bridge() as bridge: # sorts out camera and microscope control
        voltages = np.array([0.95, 0.9507, 0.9514, 2.25, 2.2513, 2.2526, 3.5, 3.5015, 3.517]) # microscope control values
        VoltageTask.ao_channels.add_ao_voltage_chan("Galvo_control/ao1")
        CameraTask.do_channels.add_do_chan("Galvo_control/port1/line3")
        core = bridge.get_core()

        if core.is_sequence_running():
            core.stop_sequence_acquisition() # stop the camera
            CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
            time.sleep(0.5/1000)
            CameraTask.write(False)

        core.start_continuous_sequence_acquisition(0) # start the camera

        CameraTask.write(True) # tell camera to take image
        time.sleep(exposure/1000)
        CameraTask.write(False)
        while core.get_remaining_image_count() == 0: #wait until picture is available
            time.sleep(0.001)
        result = core.get_last_tagged_image() # get image data into python        
        while True:

            status = stop_signal.get()
            if status == False:
                break
            else:
                stop_signal.put(True)
                if stack.empty():
                    for i in range(9):
                        VoltageTask.write(voltages[i]) # move microscope
                        time.sleep(0.05)     

                        CameraTask.write(True) # start acquisition
                        time.sleep(exposure/1000)
                        CameraTask.write(False)
                        while core.get_remaining_image_count() == 0: # wait until image is available
                            time.sleep(0.001)                
                        result = core.get_last_tagged_image() # get image data into python
                        pixels[:,:,i] = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # reshape image data
                    stack.put(pixels)
                    
        core.stop_sequence_acquisition() # stop the camera
        CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
        time.sleep(0.5/1000)
        CameraTask.write(False)
        stack.put(False)

## Start live View
def live_view(stop_signal,output,exposure,opto,x1,y1,x2,y2):
    processes = [] # initialise processes 
    proc_live = mp.Process(target=live_loop, args=(stop_signal,output,exposure,opto,x1,y1,x2,y2))
    processes.append(proc_live)

    processes.reverse()
    for process in processes:
        process.start()

    for process in processes:
        process.join()

## Start live ML-SIM
def live_ml_sim(stack,stop_signal,output,exposure):
    processes = [] # initialise processes 
    proc_live = mp.Process(target=acquisition_loop, args=(stop_signal,stack,exposure))
    processes.append(proc_live)
    proc_recon = mp.Process(target=ml_reconstruction, args=(stack,output))
    processes.append(proc_recon)
    processes.reverse()

    for process in processes:
        process.start()
    for process in processes:
        process.join()        
   
   
   
