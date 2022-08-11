import nidaqmx # microscope control
import numpy as np
import time
import torch.multiprocessing as mp
from pycromanager import Bridge
import torch
from models import *
import argparse
import serial
from tkinter.messagebox import showinfo

l405_serialport = 'COM8'
l647_serialport = 'COM5'
l488_serialport = 'COM7'
asi_stage_serialport = 'COM3'
l561_aotf_port = 'COM9'
l405_baudrate = 19200
l647_baudrate = 115200
l488_baudrate = 115200
asi_stage_baudrate = 115200
l561_aotf_baudrate = 19200

ML_looptime = 1

mp.freeze_support() 

## Grt ML-SIM params
def GetParams():
  opt = argparse.Namespace()

  # data
  opt.weights = '0216_SIMRec_0214_rndAll_rcan_continued.pth' 
  
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
  opt.n_feats = 48
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
    net = GetModel(opt) # build empty model
    print('loading checkpoint')
    checkpoint = torch.load(opt.weights,map_location=torch.device('cuda')) # load the weights
    if type(checkpoint) is dict: # remove junk pytorch stuff
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    net.module.load_state_dict(state_dict) # put weights into model
    return net


## ML-SIM reconstruction
def ml_reconstruction(stack,output,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B):
    net = load_model()
    while True:
        with torch.no_grad(): # set to eval mode
            if not stack.empty(): # waits until 9 imnages are available
                pixels = stack.get() # get those 9 images
                if isinstance(pixels, bool): # stop if sent bool false
                        break # stop reconstructing
                else:
                    pixels = pixels.astype(float)
                    iMax = rchild_max.value 
                    iMin = rchild_min.value
                    # run the reconstruction function 
                    if opto == 1:
                        result = np.zeros([512,512,3])
                        if R ==1:
                            data = torch.from_numpy(pixels) # turns pixel array into pytorh tensor
                            data = data.cuda() # move pytorch tensor to GPU
                            temp = data[y1:y1+512,x1:x1+512,:] # Crop to region of interest
                            temp = torch.swapaxes(temp,0,2) # move axes
                            temp = temp.unsqueeze(0) # pytorch needs (1,9,512,512) array or images
                            temp = temp.type(torch.FloatTensor) #  want to use float for precision
                            temp = torch.clamp(temp,iMin,iMax) # match input to traing data histogram
                            temp = temp - torch.amin(temp) # normlise between 1 and 0
                            temp = temp/torch.amax(temp) # normlise between 1 and 0
                            sr = net(temp.cuda()) # move temp array to GPU then perform reconstruction 
                            sr = torch.clamp(sr,0,1) # normalise network output
                            srframe = sr.cpu() # move output to cpu
                            srframe = srframe.numpy() # convert back to numpy array
                            result[:,:,0] = np.squeeze(srframe) # add reconstruction to stack of reconstructons (one per colour)
                        if G ==1:
                            temp = data[y2:y2+512,x2:x2+512,:]
                            temp = torch.swapaxes(temp,0,2)
                            temp = temp.unsqueeze(0)
                            temp = temp.type(torch.FloatTensor)
                            temp = torch.clamp(temp,iMin,iMax)
                            temp = temp - torch.amin(temp)
                            temp = temp/torch.amax(temp)
                            sr = net(temp.cuda())
                            sr = torch.clamp(sr,0,1)
                            srframe = sr.cpu()
                            srframe = srframe.numpy()
                            result[:,:,1] = np.squeeze(srframe)
                        if B ==1:
                            temp = data[y3:y3+512,x3:x3+512,:]
                            temp = torch.swapaxes(temp,0,2)
                            temp = temp.unsqueeze(0)
                            temp = temp.type(torch.FloatTensor)
                            temp = torch.clamp(temp,iMin,iMax)
                            temp = temp - torch.amin(temp)
                            temp = temp/torch.amax(temp)
                            sr = net(temp.cuda())
                            sr = torch.clamp(sr,0,1)
                            srframe = sr.cpu()
                            srframe = srframe.numpy()
                            result[:,:,2] = np.squeeze(srframe)
                                                                        
                        output.put(result) # sends reconstructions to plotting function    

                    else:                    
                        data = torch.from_numpy(pixels)
                        data = torch.swapaxes(data,0,2)
                        data = data.unsqueeze(0)
                        data = data.type(torch.FloatTensor)
                        data = data.cuda()
                        data = torch.clamp(data,iMin,iMax)
                        data = data - torch.amin(data)
                        data = data/torch.amax(data)
                        sr = net(data.cuda())
                        sr = torch.clamp(sr,0,1)
                        sr = torch.squeeze(sr)
                        sr = torch.swapaxes(sr,0,1)
                        sr = sr.cpu()
                        
                        sr_frame = sr.numpy()
                        sr_frame = np.squeeze(sr_frame) 
                        output.put(sr_frame)
    
    output.put(False)


## 561 Control 
def laser_control(laservariable):
    print('Inside 567 Laser')

    with nidaqmx.Task() as LaserOperationTask, nidaqmx.Task() as LaserStatusTask:
      LaserOperationTask.do_channels.add_do_chan("Dev1/port0/line0")
      print('Laser 567 port added')
      if laservariable:
        print('Turning On 561 Laser')
        LaserOperationTask.write(True)
      else:
        print('Turning Off 561 Laser')
        LaserOperationTask.write(False)

## 488 Control
def laser2_control(laservariable):
    print('Inside 488 Laser')

    with nidaqmx.Task() as LaserOperationTask, nidaqmx.Task() as LaserStatusTask:
      LaserOperationTask.do_channels.add_do_chan("Dev1/port0/line1")
      print('Laser 488 port added')
      if laservariable:
        print('Turning On 488 Laser')
        LaserOperationTask.write(True)
      else:
        print('Turning Off 488 Laser')
        LaserOperationTask.write(False)

#561 Power Control
def laserAOTF_power_control(laservariable, laserpower):

    laserAOTFSer = serial.Serial(l561_aotf_port,l561_aotf_baudrate)
    laserAOTFSer.bytesize = serial.EIGHTBITS # bits per byte
    laserAOTFSer.parity = serial.PARITY_NONE
    laserAOTFSer.stopbits = serial.STOPBITS_ONE
    laserAOTFSer.timeout = 5
    laserAOTFSer.xonxoff = False #disable software flow conrol
    laserAOTFSer.rtscts = False #disable hardware (RTS/CTS) flow control
    laserAOTFSer.dsrdtr =  False #disable hardware (DSR/DTR) flow control
    laserAOTFSer.writeTimeout = 0 #timeout for write
    laserCalcPower = float(int(laserpower)*22.5/0.065) # scaling power to AOTF scale
    laserFreq = 93.111

    print('Starting AOTF Communication for power control')


    if laservariable:
        try:
            laserAOTFSer.open()
        except Exception as e:
            print('Exception: Opening AOTF serial port: '+ str(e))
    
        if laserAOTFSer.isOpen():
            laserCommand = 'P' + str(laserCalcPower) +'\n'
            laserFreqCommand = 'F' + str(laserFreq) +'\n'
            laserAOTFSer.write(str.encode(laserFreqCommand)) 
            laserAOTFSer.write(str.encode(laserCommand)) 

            print("561 AOTF Power" +laserCommand)

            try:
                laserResponse = laserAOTFSer.readline().decode('ascii')
                print("response data: " +laserResponse)
                laserAOTFSer.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('561 AOTF Power Set')
        else:
            print('Connection failure')
    else:
        try:
            laserAOTFSer.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))

        if laserAOTFSer.isOpen():
            try:
                laserResponse = laserAOTFSer.readline().decode('ascii')
                print("response data: " +laserResponse)

                showinfo( 
                title='Warning', message=f'Laser 561 AOTF is Off - Turn On Laser to change power!'
                )
                laserAOTFSer.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('561 Laser Off')
        else:
            print('Connection failure')

#488 Power Control
def laser2_power_control(laservariable, laserpower):

    laser488Ser = serial.Serial(l488_serialport,l488_baudrate)
    laser488Ser.bytesize = serial.EIGHTBITS # bits per byte
    laser488Ser.parity = serial.PARITY_NONE
    laser488Ser.stopbits = serial.STOPBITS_ONE
    laser488Ser.timeout = 5
    laser488Ser.xonxoff = False #disable software flow conrol
    laser488Ser.rtscts = False #disable hardware (RTS/CTS) flow control
    laser488Ser.dsrdtr =  False #disable hardware (DSR/DTR) flow control
    laser488Ser.writeTimeout = 0 #timeout for write
    laserCalcPower = float(int(laserpower)/200)

    print('Starting RS-232 Communication Setup for power')


    if laservariable:
        try:
            laser488Ser.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))
    
        if laser488Ser.isOpen():
            laserCommand = 'p ' + str(laserCalcPower) +'\r'
            laser488Ser.write(str.encode(laserCommand)) 

            print("488 Power" + laserCommand)
            try:
                laserResponse = laser488Ser.readline().decode('ascii')
                print("response data: " +laserResponse)
                laser488Ser.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('488 Power Set')
        else:
            print('Connection failure')
    else:
        try:
            laser488Ser.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))

        if laser488Ser.isOpen():
            try:
                laserResponse = laser488Ser.readline().decode('ascii')
                print("response data: " +laserResponse)

                showinfo( 
                title='Warning', message=f'Laser 488 is Off - Turn On Laser to change power!'
                )
                laser488Ser.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('488 Laser Off')
        else:
            print('Connection failure')

## 405 RS-232 Control
def laser3_control(laservariable):
    print('Inside 405 Laser')

    laserSer = serial.Serial(l405_serialport,l405_baudrate)
    laserSer.bytesize = serial.EIGHTBITS # bits per byte
    laserSer.parity = serial.PARITY_NONE
    laserSer.stopbits = serial.STOPBITS_ONE
    laserSer.timeout = 5
    laserSer.xonxoff = False #disable software flow conrol
    laserSer.rtscts = False #disable hardware (RTS/CTS) flow control
    laserSer.dsrdtr =  False #disable hardware (DSR/DTR) flow control
    laserSer.writeTimeout = 0 #timeout for write

    print('Starting RS-232 Communication Setup')


    if laservariable:
        try:
            laserSer.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))
    
        if laserSer.isOpen():
            #laserSer.flushInput()
            #laserSer.flushOutput()

            laserSer.write(str.encode('L 1\r\n'))  
            print("405 On command written")
            
            try:
                laserResponse = laserSer.readline().decode('ascii')
                print("response data: " +laserResponse)
                laserSer.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('405 Laser On')
        else:
            print('Connection failure')
    else:
        try:
            laserSer.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))

        if laserSer.isOpen():
            #laserSer.flushInput()
            #laserSer.flushOutput()

            laserSer.write(str.encode('L 0\r\n')) 
            print("405 Off command written")
            try:
                laserResponse = laserSer.readline().decode('ascii')
                print("response data: " +laserResponse)
                laserSer.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('405 Laser Off')
        else:
            print('Connection failure')

def laser3_power_control(laservariable, laserpower):

    laserSer = serial.Serial(l405_serialport,l405_baudrate)
    laserSer.bytesize = serial.EIGHTBITS # bits per byte
    laserSer.parity = serial.PARITY_NONE
    laserSer.stopbits = serial.STOPBITS_ONE
    laserSer.timeout = 5
    laserSer.xonxoff = False #disable software flow conrol
    laserSer.rtscts = False #disable hardware (RTS/CTS) flow control
    laserSer.dsrdtr =  False #disable hardware (DSR/DTR) flow control
    laserSer.writeTimeout = 0 #timeout for write

    print('Starting RS-232 Communication Setup for power')


    if laservariable:
        try:
            laserSer.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))
    
        if laserSer.isOpen():
            laserCommand = 'P ' + str(laserpower) +'\n'
            laserSer.write(str.encode(laserCommand)) 

            print("405 Power" + laserCommand)
            try:
                laserResponse = laserSer.readline().decode('ascii')
                print("response data: " +laserResponse)
                laserSer.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('405 Power Set')
        else:
            print('Connection failure')
    else:
        try:
            laserSer.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))

        if laserSer.isOpen():
            try:
                laserResponse = laserSer.readline().decode('ascii')
                print("response data: " +laserResponse)

                showinfo( 
                title='Warning', message=f'Laser 405 is Off - Turn On Laser to change power!'
                )
                laserSer.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('405 Laser Off')
        else:
            print('Connection failure')

## 647 RS-232 Control
def laser4_control(laservariable):
    print('Inside 647 Laser')

    laser647Ser = serial.Serial(l647_serialport,l647_baudrate)
    laser647Ser.bytesize = serial.EIGHTBITS # bits per byte
    laser647Ser.parity = serial.PARITY_NONE
    laser647Ser.stopbits = serial.STOPBITS_ONE
    laser647Ser.timeout = 5
    laser647Ser.xonxoff = False #disable software flow conrol
    laser647Ser.rtscts = False #disable hardware (RTS/CTS) flow control
    laser647Ser.dsrdtr =  False #disable hardware (DSR/DTR) flow control
    laser647Ser.writeTimeout = 0 #timeout for write

    print('Starting RS-232 Communication Setup')


    if laservariable:
        try:
            laser647Ser.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))
    
        if laser647Ser.isOpen():
            laser647Ser.write(str.encode('en 1\r\n'))
            laser647Ser.write(str.encode('la on\r\n')) 

            print("647 On command written")
            try:
                laserResponse = laser647Ser.readline().decode('ascii')
                print("response data: " +laserResponse)
                laser647Ser.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('647 Laser On')
        else:
            print('Connection failure')
    else:
        try:
            laser647Ser.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))

        if laser647Ser.isOpen():
            laser647Ser.write(str.encode('la off\r\n')) 
            print("647 Off command written")
            try:
                laserResponse = laser647Ser.readline().decode('ascii')
                print("response data: " +laserResponse)
                laser647Ser.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('647 Laser Off')
        else:
            print('Connection failure')

#647 Power Control
def laser4_power_control(laservariable, laserpower):

    laser647Ser = serial.Serial(l647_serialport,l647_baudrate)
    laser647Ser.bytesize = serial.EIGHTBITS # bits per byte
    laser647Ser.parity = serial.PARITY_NONE
    laser647Ser.stopbits = serial.STOPBITS_ONE
    laser647Ser.timeout = 5
    laser647Ser.xonxoff = False #disable software flow conrol
    laser647Ser.rtscts = False #disable hardware (RTS/CTS) flow control
    laser647Ser.dsrdtr =  False #disable hardware (DSR/DTR) flow control
    laser647Ser.writeTimeout = 0 #timeout for write

    print('Starting RS-232 Communication Setup for 647 power')
    
    laserCalcPower = float(int(laserpower))

    if laservariable:
        try:
            laser647Ser.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))
    
        if laser647Ser.isOpen():
            laser647Ser.write(str.encode('en 1\r\n'))
            laserCommand = 'ch 1 pow ' + str(laserCalcPower) +'\n'
            laser647Ser.write(str.encode(laserCommand)) 

            print("647 Power" + laserCommand)
            try:
                laserResponse = laser647Ser.readline().decode('ascii')
                print("response data: " +laserResponse)
                laser647Ser.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('647 Power Set')
        else:
            print('Connection failure')
    else:
        try:
            laser647Ser.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))

        if laser647Ser.isOpen():
            try:
                laserResponse = laser647Ser.readline().decode('ascii')
                print("response data: " +laserResponse)

                showinfo( 
                title='Warning', message=f'Laser 647 is Off - Turn On Laser to change power!'
                )
                laser647Ser.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('647 Laser Off')
        else:
            print('Connection failure')

## Z axis control testing for OS-SIM
def z_axis_control(zvalue, zstatus):

    ASI_StageSer = serial.Serial(asi_stage_serialport,asi_stage_baudrate)
    ASI_StageSer.bytesize = serial.EIGHTBITS # bits per byte
    ASI_StageSer.parity = serial.PARITY_NONE
    ASI_StageSer.stopbits = serial.STOPBITS_ONE
    ASI_StageSer.timeout = 5
    ASI_StageSer.xonxoff = False #disable software flow conrol
    ASI_StageSer.rtscts = False #disable hardware (RTS/CTS) flow control
    ASI_StageSer.dsrdtr =  False #disable hardware (DSR/DTR) flow control
    ASI_StageSer.writeTimeout = 0 #timeout for write

    print('Starting RS-232 Communication Setup for ASI Stage Control')
    
    z_axis_value = float(int(zvalue))

    if zstatus:
        try:
            ASI_StageSer.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))
    
        if ASI_StageSer.isOpen():
            ASI_StageSer.write(str.encode('en 1\r\n'))
            ASI_ZCommand = 'R X Y Z= ' + str(z_axis_value) +'\n'
            ASI_StageSer.write(str.encode(ASI_ZCommand)) 

            print("AZI Z Command" + ASI_ZCommand)
            try:
                ASI_StageResponse = ASI_StageSer.readline().decode('ascii')
                print("response data: " +ASI_StageResponse)
                ASI_StageSer.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('ASi Stage Z axis moved')
        else:
            print('Connection failure')
    else:
        try:
            ASI_StageSer.open()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))

        if ASI_StageSer.isOpen():
            try:
                ASI_StageResponse = ASI_StageSer.readline().decode('ascii')
                print("response data: " +ASI_StageResponse)

                showinfo( 
                title='Warning', message=f'AZI Stage Control is Off - Turn On Device for Control!'
                )
                ASI_StageSer.flush()
            except Exception as e:
                print('Exception: Writing to serial port: '+ str(e))
            print('ASI Stage control off')
        else:
            print('Connection failure')

## Live view
def live_loop(stop_signal,output,exposure,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B):
    print('starting acquisition')
    
    stop_signal.put(True)
    with nidaqmx.Task() as VoltageTask, nidaqmx.Task() as CameraTask, Bridge() as bridge: # sorts out camera and microscope control
        voltages = np.array([0.95, 0.9508, 0.9516, 2.25, 2.2025, 2.205, 3.45, 3.454, 3.4548]) # microscope control values
        waits = np.array([0.08,0.008,0.008,0.06,0.008,0.008,0.06,0.008,0.008])
        VoltageTask.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        CameraTask.do_channels.add_do_chan("Dev1/port0/line2")
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
                        time.sleep(waits[i])   

                        #print('Voltages :',voltages[i],'Wait Time :' , waits[i])

                        CameraTask.write(True) # start acquisition
                        time.sleep(exposure/1000)
                        CameraTask.write(False)
                        while core.get_remaining_image_count() == 0: # wait until image is available
                            time.sleep(0.001)                
                        result = core.get_last_tagged_image() # get image data into python
                        pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # reshape image data
                        pixels = pixels.astype('float64')
                        if opto == 1:
                            merged = np.zeros([512,512,3])
                            if R ==1:
                                q = pixels[y1:y1+512,x1:x1+512]
                                q = q/np.amax(q)
                                merged[:,:,0] = q
                                
                            if G ==1:    
                                q = pixels[y2:y2+512,x2:x2+512]
                                q = q/np.amax(q)
                                merged[:,:,1] = q

                            if B ==1:
                                q = pixels[y3:y3+512,x3:x3+512]
                                q = q/np.amax(q)
                                merged[:,:,2] = q
                            output.put(merged)
                        else:      
                            iMax = rchild_max.value 
                            iMin = rchild_min.value
                            pixels = np.clip(pixels, iMin,iMax)
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
    
    with nidaqmx.Task() as VoltageTask, nidaqmx.Task() as CameraTask, Bridge() as bridge: # sorts out camera and microscope control
        voltages = np.array([0.95, 0.9508, 0.9516, 2.25, 2.2025, 2.205, 3.45, 3.454, 3.4548]) # microscope control values
        waits = np.array([0.08,0.008,0.008,0.06,0.008,0.008,0.06,0.008,0.008])
        VoltageTask.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        CameraTask.do_channels.add_do_chan("Dev1/port0/line2")
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
                        time.sleep(waits[i])     

                        CameraTask.write(True) # start acquisition
                        time.sleep(exposure/1000)
                        CameraTask.write(False)

                        while core.get_remaining_image_count() == 0: # wait until image is available
                            time.sleep(0.001)                
                        result = core.get_last_tagged_image() # get image data into python
                        pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # reshape image data
                        if i == 0:
                            merged = np.zeros([pixels.shape[0],pixels.shape[1],9])
                            merged[:,:,0] = pixels
                        else:
                            merged[:,:,i] = pixels

                    stack.put(merged)
                    
        core.stop_sequence_acquisition() # stop the camera
        CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
        time.sleep(0.5/1000)
        CameraTask.write(False)
        stack.put(False)

## Start live View
def live_view(stop_signal,output,exposure,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B):
    processes = [] # initialise processes 
    proc_live = mp.Process(target=live_loop, args=(stop_signal,output,exposure,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))
    processes.append(proc_live)

    processes.reverse()
    for process in processes:
        process.start()

    for process in processes:
        process.join()

## Start live ML-SIM
def live_ml_sim(stack,stop_signal,output,exposure,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B):
    processes = [] # initialise processes 
    proc_live = mp.Process(target=acquisition_loop, args=(stop_signal,stack,exposure))
    processes.append(proc_live)
    proc_recon = mp.Process(target=ml_reconstruction, args=(stack,output,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))
    processes.append(proc_recon)
    processes.reverse()

    for process in processes:
        process.start()
    for process in processes:
        process.join()   


## Live view - Updated
def live_loop_v2(stop_signal,output,exposure,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B):
    print('starting acquisition V2')
    looptime = 1
    startloop = 0
    stoploop = 8
    step = 1
    
    stop_signal.put(True)
    with nidaqmx.Task() as VoltageTask, nidaqmx.Task() as CameraTask, Bridge() as bridge: # sorts out camera and microscope control
       # voltages = np.array([0.95, 0.9508, 0.9516, 2.25, 2.2025, 2.205, 3.45, 3.454, 3.4548, 2.25, 2.2025, 2.205]) # microscope control values
       # waits = np.array([0.06,0.008,0.008,0.06,0.008,0.008,0.06,0.008,0.008,0.06,0.008,0.008])
        voltages = np.array([0.95, 0.9508, 0.9516, 2.25, 2.2025, 2.205, 3.45, 3.454, 3.4548]) # microscope control values
        waits = np.array([0.06,0.008,0.008,0.06,0.008,0.008,0.06,0.008,0.008])
        VoltageTask.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        CameraTask.do_channels.add_do_chan("Dev1/port0/line2")
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

                    if (looptime % 2==0):
                        startloop=8
                        stoploop=-1
                        step = -1
                    else: 
                        startloop=0
                        stoploop=8
                        step = 1

                    print('looptime :',looptime)
                    print('start time :',startloop)
                    print('end time :',stoploop)

                    for i in range(startloop,stoploop,step):
                        VoltageTask.write(voltages[i]) # move microscope
                        time.sleep(waits[i])

                        print('Voltage :',voltages[i],'Wait Time :' , waits[i])      

                        CameraTask.write(True) # start acquisition
                        time.sleep(exposure/1000)
                        CameraTask.write(False)
                        while core.get_remaining_image_count() == 0: # wait until image is available
                            time.sleep(0.001)                
                        result = core.get_last_tagged_image() # get image data into python
                        pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # reshape image data
                        pixels = pixels.astype('float64')
                        if opto == 1:
                            merged = np.zeros([512,512,3])
                            if R ==1:
                                q = pixels[y1:y1+512,x1:x1+512]
                                q = q/np.amax(q)
                                merged[:,:,0] = q
                                
                            if G ==1:    
                                q = pixels[y2:y2+512,x2:x2+512]
                                q = q/np.amax(q)
                                merged[:,:,1] = q

                            if B ==1:
                                q = pixels[y3:y3+512,x3:x3+512]
                                q = q/np.amax(q)
                                merged[:,:,2] = q
                            output.put(merged)
                        else:      
                            iMax = rchild_max.value 
                            iMin = rchild_min.value
                            pixels = np.clip(pixels, iMin,iMax)
                            output.put(pixels)

                    looptime = looptime+1
                   
        core.stop_sequence_acquisition() # stop the camera
        CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
        time.sleep(0.5/1000)
        CameraTask.write(False)
        output.put(False)

## Live aquisition loop - Updated
def acquisition_loop_v2(stop_signal,stack,exposure):
    print('starting acquisition')
    ML_looptime = 1
    startloop = 0
    stoploop = 8
    step = 1

    stop_signal.put(True)
    
    with nidaqmx.Task() as VoltageTask, nidaqmx.Task() as CameraTask, Bridge() as bridge: # sorts out camera and microscope control
        voltages = np.array([0.95, 0.9508, 0.9516, 2.25, 2.2025, 2.205, 3.45, 3.454, 3.4548]) # microscope control values
        waits = np.array([0.06,0.008,0.008,0.06,0.008,0.008,0.06,0.008,0.008])
        VoltageTask.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        CameraTask.do_channels.add_do_chan("Dev1/port0/line2")
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

                    if (ML_looptime % 2==0):
                        startloop=8
                        stoploop=-1
                        step = -1
                    else: 
                        startloop=0
                        stoploop=8
                        step = 1

                    print('looptime :',ML_looptime)
                    print('start time :',startloop)
                    print('end time :',stoploop)

                    for i in range(startloop,stoploop,step):
                        VoltageTask.write(voltages[i]) # move microscope
                        time.sleep(waits[i])  

                        print('Voltages :',voltages[i],'Wait Time :' , waits[i])       

                        CameraTask.write(True) # start acquisition
                        time.sleep(exposure/1000)
                        CameraTask.write(False)

                        while core.get_remaining_image_count() == 0: # wait until image is available
                            time.sleep(0.001)                
                        result = core.get_last_tagged_image() # get image data into python
                        pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # reshape image data
                        if (i == 0 or i==-1):
                            merged = np.zeros([pixels.shape[0],pixels.shape[1],9])
                            merged[:,:,0] = pixels
                        elif(i>0 and i<=8):
                            merged[:,:,i] = pixels
                        else:
                            print('i value',i)
                    stack.put(merged)
                    
                    ML_looptime = ML_looptime +1

                    
        core.stop_sequence_acquisition() # stop the camera
        CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
        time.sleep(0.5/1000)
        CameraTask.write(False)
        stack.put(False)

## Start live View Updated
def live_view_v2(stop_signal,output,exposure,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B):
    print('Running updated version live view')
    processes = [] # initialise processes 
    proc_live = mp.Process(target=live_loop_v2, args=(stop_signal,output,exposure,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))
    processes.append(proc_live)

    processes.reverse()
    for process in processes:
        process.start()

    for process in processes:
        process.join()

## Start live ML-SIM Updtaed 
def live_ml_sim_v2(stack,stop_signal,output,exposure,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B):#
    print('Running updated version live ml sim')
    processes = [] # initialise processes 
    proc_live = mp.Process(target=acquisition_loop_v2, args=(stop_signal,stack,exposure))
    processes.append(proc_live)
    proc_recon = mp.Process(target=ml_reconstruction, args=(stack,output,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))
    processes.append(proc_recon)
    processes.reverse()

    for process in processes:
        process.start()
    for process in processes:
        process.join()      
   
   
   
