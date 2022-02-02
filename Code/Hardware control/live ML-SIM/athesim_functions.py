import nidaqmx # microscope control
import numpy as np
import time
import torch.multiprocessing as mp
from pycromanager import Bridge
import torch
from models import *
import argparse
from swinir_arch import SwinIR

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

## Grt ML-SIM params
def get_VSR_params():
  opt = argparse.Namespace()

  # input/output layer options
  opt.task = 'simrec'
  opt.patch_size = 64
  opt.scale = 2
  opt.noise = 0
  opt.jpeg = 40
  opt.large_model = True
  opt.model_path = 'C:/Users/SIM_ADMIN/Downloads/VSR/SwinIR_RCAB_model-opts-20220202/experiments/SwinIR_RCAB/net_g_latest.pth'

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

## Load VSR model
def load_VSR_model():
    opts = get_VSR_params()
    device = torch.device('cuda')
    model = define_model(opts)
    model.eval()
    model = model.to(device)

    return model

def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = SwinIR(
            upscale=args.scale,
            in_chans=3,
            img_size=args.patch_size,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv')

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = SwinIR(
            upscale=args.scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffledirect',
            resi_connection='1conv')

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='nearest+conv',
                resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=248,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2,
                upsampler='nearest+conv',
                resi_connection='3conv')

    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = SwinIR(
            upscale=1,
            in_chans=1,
            img_size=128,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv')

    # 005 color image denoising
    elif args.task == 'color_dn':
        model = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=128,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv')

    # 006 JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's slightly better than 1
    elif args.task == 'jpeg_car':
        model = SwinIR(
            upscale=1,
            in_chans=1,
            img_size=126,
            window_size=7,
            img_range=255.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv')
    elif args.task == 'simrec':
        model = SwinIR(
            upscale=2,
            in_chans=9,
            img_size=512,
            window_size=4,
            img_range=1,
            depths=[6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv',
            vis=True,
            **{'pixelshuffleFactor':2})


    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model

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
                    data = data - np.amin(data)
                    data = data/np.amax(data)
                    sr = net(data.cuda())

                    sr = sr.cpu()
                    sr_frame = sr.numpy()
                    sr_frame = np.squeeze(sr_frame) 
                    output.put(sr_frame)
    
    output.put(False)

## Live view
def live_loop(stop_signal,output,exposure):
    print('starting acquisition')
    stop_signal.put(True)
    with nidaqmx.Task() as VoltageTask, nidaqmx.Task() as CameraTask, Bridge() as bridge: # sorts out camera and microscope control
        voltages = np.array([0.95, 0.9507, 0.9514, 2.25, 2.2513, 2.2526, 3.5, 3.5015, 3.517]) # microscope control values
        VoltageTask.ao_channels.add_ao_voltage_chan("Galvo_control/ao0")
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
        VoltageTask.ao_channels.add_ao_voltage_chan("Galvo_control/ao0")
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
def live_view(stop_signal,output,exposure):
    processes = [] # initialise processes 
    proc_live = mp.Process(target=live_loop, args=(stop_signal,output,exposure))
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
    
   
if __name__ == '__main__':
    print('Done')
    model = load_VSR_model()   
   
   
   
   
   
   
